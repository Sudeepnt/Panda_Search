import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { ConfigService } from '../config';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { execFile } from 'child_process';
import { promisify } from 'util';
import * as sharp from 'sharp';

const execFileAsync = promisify(execFile);

// 512px keeps enough visual/OCR detail for file search while making a whole
// Finder library materially faster on CPU-only Apple-silicon inference.
const MAX_IMAGE_DIMENSION = 512;
// sips is only used when sharp cannot decode a Finder-native format. Keep its
// raster output slightly smaller because square app icons otherwise consume
// more than a 1K-token llama.cpp parallel slot before generation begins.
const SIPS_MAX_IMAGE_DIMENSION = 448;
const MAX_IMAGE_BYTES = 25 * 1024 * 1024; // 25MB
const RETRY_ATTEMPTS = 2;
const RETRY_DELAY_MS = 2000;
const RECOVERY_COOLDOWN_MS = 60000; // Re-check the local vision model after 1 min

export interface ImageIndexDescription {
  title: string;
  explanation: string;
  searchableText: string;
}

@Injectable()
export class ImageCaptioningService implements OnModuleInit {
  private readonly logger = new Logger(ImageCaptioningService.name);
  // Qwen is bundled locally and served by llama.cpp. Do not route visual
  // indexing through Ollama: that service may be used for chat, but it does
  // not own Panda's vision model.
  private readonly modelName = 'Qwen3VL-4B-Instruct-Q4_K_M.gguf';
  private readonly visionUrl = process.env.PANDA_VISION_URL || 'http://127.0.0.1:8081';
  private isVisionModelAvailable = false;
  private lastVisionModelFailureTime = 0;

  constructor(private readonly configService: ConfigService) {
  }

  async onModuleInit() {
    await this.checkVisionModelHealth();
  }

  /**
   * Check whether Panda's local Qwen-VL server is available.
   */
  async checkVisionModelHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.visionUrl}/v1/models`);
      if (!response.ok) {
        this.logger.warn('Panda Qwen-VL server is not responding');
        this.isVisionModelAvailable = false;
        return false;
      }

      const data = await response.json();
      const models = data.data || [];
      const hasVisionModel = models.some((model: { id: string }) =>
        typeof model.id === 'string' && model.id.toLowerCase().includes('qwen3vl'),
      );

      if (hasVisionModel) {
        this.logger.log(`Vision model ${this.modelName} is available`);
        this.isVisionModelAvailable = true;
      } else {
        this.logger.warn(
          'Panda Qwen-VL model was not reported by the local server.',
        );
        this.isVisionModelAvailable = false;
      }

      return this.isVisionModelAvailable;
    } catch (error: any) {
      this.logger.warn(
        `Failed to check Panda Qwen-VL: ${error.message}`,
      );
      this.isVisionModelAvailable = false;
      return false;
    }
  }

  /**
   * Generate a detailed caption for an image using the local vision model.
   * @param imagePath - Path to the image file
   * @returns Detailed text description of the image
   */
  async generateCaption(imagePath: string): Promise<string> {
    return (await this.generateImageIndexDescription(imagePath)).searchableText;
  }

  /**
   * One VLM pass used by the app's persistent image index.  The structured
   * values are shown in the table; searchableText is stored in LanceDB.
   */
  async generateImageIndexDescription(imagePath: string): Promise<ImageIndexDescription> {
    // Verify file exists
    if (!fs.existsSync(imagePath)) {
      throw new Error(`Image file not found: ${imagePath}`);
    }

    // Re-check health if previously failed and cooldown has passed
    if (
      !this.isVisionModelAvailable &&
      Date.now() - this.lastVisionModelFailureTime > RECOVERY_COOLDOWN_MS
    ) {
      await this.checkVisionModelHealth();
    }

    // Check health if not already available
    if (!this.isVisionModelAvailable) {
      await this.checkVisionModelHealth();
    }

    // If still not available, use fallback
    if (!this.isVisionModelAvailable) {
      this.logger.warn(
        `Vision model not available, using filename fallback for: ${imagePath}`,
      );
      return this.getFallbackImageIndexDescription(imagePath);
    }

    try {
      // Read and optionally resize image to reduce local inference memory.
      const imageBuffer = await this.prepareImageForCaptioning(imagePath);
      const base64Image = imageBuffer.toString('base64');

      this.logger.log(`Generating caption for: ${path.basename(imagePath)}`);

      let lastError: Error | null = null;
      for (let attempt = 0; attempt <= RETRY_ATTEMPTS; attempt++) {
        try {
          const response = await fetch(
            `${this.visionUrl}/v1/chat/completions`,
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                model: this.modelName,
                temperature: 0.1,
                // Captions are intentionally compact: the table needs a
                // useful title plus searchable visual facts, not a long
                // essay. Keeping the cap bounded makes a whole-library scan
                // responsive on Apple silicon.
                max_tokens: 128,
                messages: [{
                  role: 'user',
                  content: [
                    { type: 'text', text: `You are indexing a personal file library. Reply in exactly two sections: first line must be TITLE: followed by a concise, factual 3–8 word title; second section must be DESCRIPTION: followed by a factual, dense visual-search description. Include scene and setting; important objects/people; screen layout and visible app content; spatial relationships; actions; colors/style; readable text/OCR; and searchable tags. Do not invent details.` },
                    { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${base64Image}` } },
                  ],
                }],
              }),
            },
          );
          if (!response.ok) throw new Error(await response.text());
          const data = await response.json();
          const rawCaption = (data.choices?.[0]?.message?.content ?? '').trim();
          const fileName = path.basename(imagePath);
          const ext = path.extname(imagePath).toLowerCase().replace('.', '');
          const parsed = this.parseImageIndexDescription(rawCaption, imagePath);
          const searchableText = `[image, picture, file, ${ext} format, ${fileName}] Title: ${parsed.title}. ${parsed.explanation}`;
          this.logger.log(
            `Generated image index (${searchableText.length} chars): ${searchableText.substring(0, 100)}...`,
          );

          return { ...parsed, searchableText };
        } catch (err: any) {
          lastError = err;
          if (attempt < RETRY_ATTEMPTS) {
            this.logger.warn(
              `Vision-model attempt ${attempt + 1} failed (${err.message}), retrying in ${RETRY_DELAY_MS}ms...`,
            );
            await this.sleep(RETRY_DELAY_MS);
          }
        }
      }

      throw lastError ?? new Error('Captioning failed');
    } catch (error: any) {
      this.logger.warn(
        `Vision captioning failed, using filename fallback: ${error.message}`,
      );

      this.isVisionModelAvailable = false;
      this.lastVisionModelFailureTime = Date.now();

      return this.getFallbackImageIndexDescription(imagePath);
    }
  }

  /**
   * Prepare image for captioning - resize and convert to JPEG to avoid Ollama OOM.
   * WebP/PNG can cause memory issues; JPEG is more reliable for Ollama.
   */
  private async prepareImageForCaptioning(imagePath: string): Promise<Buffer> {
    const imageBuffer = await fs.promises.readFile(imagePath);
    const ext = path.extname(imagePath).toLowerCase();

    try {
      // sharp intentionally does not decode Windows ICO resources. macOS's
      // built-in sips converter does, so normalize these tiny Finder icons to
      // JPEG before sending them to Qwen instead of leaving a filename-only
      // fallback row in the visual index.
      if (ext === '.ico') {
        const tempDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'panda-vlm-'));
        const sourcePath = path.join(tempDir, 'source.ico');
        const outputPath = path.join(tempDir, 'image.jpg');
        try {
          await fs.promises.writeFile(sourcePath, imageBuffer);
          await execFileAsync('/usr/bin/sips', [
            '-Z', String(SIPS_MAX_IMAGE_DIMENSION),
            '-s', 'format', 'jpeg', sourcePath, '--out', outputPath,
          ]);
          const converted = await fs.promises.readFile(outputPath);
          this.logger.log(
            `Prepared ICO for captioning: ${imageBuffer.length} -> ${converted.length} bytes`,
          );
          return converted;
        } finally {
          await fs.promises.rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
        }
      }

      const metadata = await sharp(imageBuffer).metadata();
      const width = metadata.width ?? 0;
      const height = metadata.height ?? 0;

      // Send a normalized JPEG to Qwen. Besides reducing memory, this makes
      // Finder formats such as SVG, ICO, ICNS, AVIF, HEIC, and TIFF readable by
      // the vision server instead of silently falling back to a filename.
      const needsConvert = !['.jpg', '.jpeg'].includes(ext);
      const needsResize =
        width > MAX_IMAGE_DIMENSION ||
        height > MAX_IMAGE_DIMENSION ||
        imageBuffer.length > MAX_IMAGE_BYTES;

      if (!needsConvert && !needsResize) {
        return imageBuffer;
      }

      let pipeline = sharp(imageBuffer);

      if (needsResize) {
        pipeline = pipeline.resize(MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION, {
          fit: 'inside',
          withoutEnlargement: true,
        });
      }

      const processed = await pipeline.jpeg({ quality: 80 }).toBuffer();

      this.logger.log(
        `Prepared image for captioning: ${imageBuffer.length} -> ${processed.length} bytes`,
      );
      return processed;
    } catch (err: any) {
      // Some real-world SVGs (notably extension walkthrough art) contain
      // malformed optional XML namespaces that sharp rejects. AVIF support
      // also varies by the libvips build bundled with the app. macOS sips is
      // more forgiving and can rasterize these Finder image formats for the
      // VLM instead of leaving a filename-only fallback row.
      if (['.svg', '.icns', '.avif', '.raw', '.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.raf', '.pef', '.srw'].includes(ext)) {
        try {
          const tempDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'panda-vlm-'));
          const sourcePath = path.join(tempDir, `source${ext}`);
          const outputPath = path.join(tempDir, 'image.jpg');
          try {
            await fs.promises.writeFile(sourcePath, imageBuffer);
            await execFileAsync('/usr/bin/sips', [
              '-Z', String(SIPS_MAX_IMAGE_DIMENSION),
              '-s', 'format', 'jpeg', sourcePath, '--out', outputPath,
            ]);
            const converted = await fs.promises.readFile(outputPath);
            this.logger.log(
              `Prepared ${ext.slice(1).toUpperCase()} with sips: ${imageBuffer.length} -> ${converted.length} bytes`,
            );
            return converted;
          } finally {
            await fs.promises.rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
          }
        } catch (sipsError: any) {
          this.logger.warn(`sips image prepare failed, using original: ${sipsError.message}`);
        }
      }
      this.logger.warn(`Image prepare failed, using original: ${err.message}`);
      return imageBuffer;
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Fallback caption when the local vision model is unavailable.
   * Extracts searchable terms from filename (e.g. "beautiful-brown-horse" -> horse, brown, beautiful).
   */
  private parseImageIndexDescription(raw: string, imagePath: string): Pick<ImageIndexDescription, 'title' | 'explanation'> {
    const fallback = this.getFallbackImageIndexDescription(imagePath);
    const titleMatch = raw.match(/^\s*TITLE\s*:\s*(.+)$/im);
    const descriptionMatch = raw.match(/DESCRIPTION\s*:\s*([\s\S]*)$/im);
    return {
      title: titleMatch?.[1].trim() || fallback.title,
      explanation: descriptionMatch?.[1].trim() || raw.replace(/^\s*TITLE\s*:\s*.*$/im, '').trim() || fallback.explanation,
    };
  }

  private getFallbackImageIndexDescription(imagePath: string): ImageIndexDescription {
    const fileName = path.basename(imagePath);
    const ext = path.extname(imagePath).toLowerCase().replace('.', '');
    const nameWithoutExt = path.basename(fileName, ext ? `.${ext}` : '');

    // Extract hyphen/underscore-separated words as searchable terms (skip numbers)
    const words = nameWithoutExt
      .split(/[-_\s.]+/)
      .filter((w) => w.length > 2 && !/^\d+$/.test(w))
      .map((w) => w.toLowerCase());

    const searchTerms =
      words.length > 0
        ? ` Search terms: ${[...new Set(words)].join(', ')}.`
        : '';

    const title = nameWithoutExt.replace(/[-_]+/g, ' ').trim() || 'Image file';
    const explanation = `Image file: ${fileName}.${searchTerms}`;
    return {
      title,
      explanation,
      searchableText: `[image, picture, file, ${ext} format, ${fileName}] Title: ${title}. ${explanation}`,
    };
  }

  /**
   * Get the current status of the service
   */
  getStatus(): { available: boolean; model: string } {
    return {
      available: this.isVisionModelAvailable,
      model: this.modelName,
    };
  }
}
