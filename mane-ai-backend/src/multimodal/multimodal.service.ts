import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ffmpeg = require('fluent-ffmpeg');
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ffmpegInstaller = require('@ffmpeg-installer/ffmpeg');

// Set ffmpeg path
ffmpeg.setFfmpegPath(ffmpegInstaller.path);

// Types for pipelines
type Pipeline = any;

interface TranscriptionResult {
  text: string;
}

export type MediaType = 'text' | 'image' | 'audio' | 'video';

export interface ProcessedMedia {
  mediaType: MediaType;
  vector: number[];
  content: string; // Text content or transcript
}

@Injectable()
export class MultimodalService implements OnModuleInit {
  private readonly logger = new Logger(MultimodalService.name);

  // Lazy-loaded pipelines (Singleton pattern)
  private whisperPipeline: Pipeline = null;
  private textPipeline: Pipeline = null;

  // Embedding dimensions
  readonly TEXT_DIMENSION = 384; // all-MiniLM-L6-v2

  // Supported file extensions
  private readonly imageExtensions = [
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.heic', '.heif', '.tiff', '.bmp',
    '.svg', '.ico', '.icns', '.avif', '.raw', '.cr2', '.cr3', '.nef', '.arw',
    '.dng', '.orf', '.rw2', '.raf', '.pef', '.srw',
  ];
  private readonly audioExtensions = ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.aiff', '.wma'];
  private readonly videoExtensions = ['.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'];
  private readonly textExtensions = [
    '.txt', '.md', '.markdown', '.csv', '.tsv', '.pdf', '.docx', '.doc', '.pages', '.odt', '.rtf', '.rtfd', '.epub', '.mobi',
    '.xlsx', '.xls', '.xlsm', '.numbers', '.ods', '.pptx', '.ppt', '.key', '.odp',
    '.json', '.jsonl', '.ndjson', '.xml', '.yaml', '.yml', '.toml', '.ini', '.conf', '.config', '.env', '.log', '.sql',
    '.html', '.htm', '.css', '.scss', '.less', '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs', '.py', '.pyw', '.swift',
    '.m', '.h', '.c', '.cc', '.cpp', '.cxx', '.hpp', '.java', '.kt', '.kts', '.go', '.rs', '.rb', '.php', '.sh', '.bash', '.zsh', '.fish', '.graphql', '.gql', '.tex',
    '.svelte', '.vue', '.astro', '.cu', '.cuh', '.wgsl', '.glsl', '.metal', '.plist', '.strings', '.stringsdata', '.cmake', '.mk', '.make', '.lock', '.properties', '.pbxproj', '.entitlements', '.ps1', '.bat', '.vim', '.nix', '.jinja', '.jinja2', '.template', '.tmpl', '.example', '.inp', '.dia', '.d', '.eml', '.msg', '.vcf', '.ics',
  ];

  async onModuleInit() {
    // Pre-load text pipeline on startup (most commonly used)
    this.logger.log('Pre-loading text embedding pipeline...');
    await this.getTextPipeline();
  }

  /**
   * Determine media type from file extension
   */
  getMediaType(filePath: string): MediaType {
    const ext = path.extname(filePath).toLowerCase();

    if (this.imageExtensions.includes(ext)) return 'image';
    if (this.audioExtensions.includes(ext)) return 'audio';
    if (this.videoExtensions.includes(ext)) return 'video';
    return 'text';
  }

  /**
   * Process any file and return embeddings
   */
  async processFile(
    filePath: string,
    content?: string,
  ): Promise<ProcessedMedia> {
    const mediaType = this.getMediaType(filePath);

    this.logger.log(`Processing ${mediaType} file: ${path.basename(filePath)}`);

    switch (mediaType) {
      case 'audio':
        return this.processAudio(filePath);
      default:
        return this.processText(filePath, content);
    }
  }

  /**
   * Process text content
   */
  async processText(
    filePath: string,
    content?: string,
  ): Promise<ProcessedMedia> {
    const textContent =
      content || (await fs.promises.readFile(filePath, 'utf-8'));

    const vector = await this.embedText(textContent);

    return {
      mediaType: 'text',
      vector,
      content: textContent,
    };
  }

  /**
   * Process audio file using Whisper for transcription
   */
  async processAudio(filePath: string): Promise<ProcessedMedia> {
    try {
      // Transcribe audio to text
      const transcript = await this.transcribeAudio(filePath);

      // Embed the transcript
      const vector = await this.embedText(transcript);

      return {
        mediaType: 'audio',
        vector,
        content: transcript,
      };
    } catch (error: any) {
      this.logger.error(`Failed to process audio: ${error.message}`);
      throw error;
    }
  }

  /** Export three points in a video so Qwen indexes more than its opening. */
  async extractVideoPreviews(filePath: string): Promise<string[]> {
    const id = `pandaintelligence_video_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    const outputDir = os.tmpdir();
    return new Promise((resolve, reject) => {
      ffmpeg(filePath)
        .on('error', (error: Error) =>
          reject(new Error(`Video frame extraction failed: ${error.message}`)),
        )
        .on('end', () => resolve(['10', '50', '90'].map((point) => path.join(outputDir, `${id}_${point}.jpg`))))
        .screenshots({
          timestamps: ['10%', '50%', '90%'],
          filename: `${id}_%i.jpg`,
          folder: outputDir,
          size: '1024x?',
        });
    });
  }

  /**
   * Get or initialize Whisper pipeline for audio transcription
   */
  private async getWhisperPipeline(): Promise<Pipeline> {
    if (!this.whisperPipeline) {
      this.logger.log('Loading Whisper pipeline (Xenova/whisper-tiny)...');
      const { pipeline } = await import('@huggingface/transformers');
      this.whisperPipeline = await pipeline(
        'automatic-speech-recognition',
        'Xenova/whisper-tiny',
      );
      this.logger.log('Whisper pipeline loaded successfully');
    }
    return this.whisperPipeline;
  }

  /**
   * Get or initialize text embedding pipeline
   */
  private async getTextPipeline(): Promise<Pipeline> {
    if (!this.textPipeline) {
      this.logger.log('Loading text pipeline (Xenova/all-MiniLM-L6-v2)...');
      const { pipeline } = await import('@huggingface/transformers');
      this.textPipeline = await pipeline(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2',
      );
      this.logger.log('Text pipeline loaded successfully');
    }
    return this.textPipeline;
  }

  /**
   * Generate text embedding using text model (384-dim)
   */
  async embedText(text: string): Promise<number[]> {
    const pipe = await this.getTextPipeline();

    const output = await pipe(text, {
      pooling: 'mean',
      normalize: true,
    });

    const embedding = Array.from(output.data) as number[];
    output.dispose?.();
    return embedding;
  }

  /**
   * Convert audio file to 16kHz mono WAV format for Whisper
   */
  private async convertToWav(inputPath: string): Promise<string> {
    const tempDir = os.tmpdir();
    const outputPath = path.join(tempDir, `whisper_${Date.now()}.wav`);

    return new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .audioFrequency(16000) // 16kHz sample rate (Whisper requirement)
        .audioChannels(1) // Mono
        .audioCodec('pcm_s16le') // 16-bit PCM
        .format('wav')
        .on('error', (err) => {
          reject(new Error(`FFmpeg conversion failed: ${err.message}`));
        })
        .on('end', () => {
          resolve(outputPath);
        })
        .save(outputPath);
    });
  }

  /**
   * Read WAV file and extract raw PCM audio as Float32Array
   */
  private async readWavAsFloat32(wavPath: string): Promise<Float32Array> {
    const buffer = await fs.promises.readFile(wavPath);

    // Parse WAV header (44 bytes for standard PCM WAV)
    // Skip to data chunk
    let dataOffset = 12; // Skip RIFF header
    while (dataOffset < buffer.length - 8) {
      const chunkId = buffer.toString('ascii', dataOffset, dataOffset + 4);
      const chunkSize = buffer.readUInt32LE(dataOffset + 4);

      if (chunkId === 'data') {
        dataOffset += 8; // Skip chunk header
        break;
      }
      dataOffset += 8 + chunkSize;
    }

    // Extract PCM data (16-bit signed integers)
    const pcmData = buffer.subarray(dataOffset);
    const samples = new Float32Array(pcmData.length / 2);

    for (let i = 0; i < samples.length; i++) {
      // Convert 16-bit signed integer to float [-1, 1]
      const int16 = pcmData.readInt16LE(i * 2);
      samples[i] = int16 / 32768.0;
    }

    return samples;
  }

  /**
   * Transcribe audio using Whisper
   * Converts audio to 16kHz mono WAV and passes raw PCM to pipeline
   * @throws Error if transcription fails - no silent fallback
   */
  async transcribeAudio(audioPath: string): Promise<string> {
    const pipe = await this.getWhisperPipeline();
    const fileName = path.basename(audioPath);
    let tempWavPath: string | null = null;

    try {
      this.logger.log(`Transcribing audio: ${fileName}`);

      // Convert to 16kHz mono WAV
      this.logger.log(`Converting ${fileName} to 16kHz WAV...`);
      tempWavPath = await this.convertToWav(audioPath);

      // Read WAV as Float32Array
      this.logger.log(`Reading PCM data from WAV...`);
      const audioData = await this.readWavAsFloat32(tempWavPath);
      const durationSec = (audioData.length / 16000).toFixed(1);
      this.logger.log(
        `Audio loaded: ${audioData.length} samples (${durationSec}s)`,
      );

      // Pass raw audio data to Whisper with chunking for long audio
      // chunk_length_s: Process audio in 30-second chunks
      // stride_length_s: Overlap chunks by 5 seconds for better continuity
      const result = (await pipe(audioData, {
        sampling_rate: 16000,
        chunk_length_s: 30,
        stride_length_s: 5,
        return_timestamps: false,
      })) as TranscriptionResult;

      const transcript = result.text?.trim() || '';

      if (!transcript) {
        throw new Error('Whisper returned empty transcript');
      }

      this.logger.log(
        `Successfully transcribed ${fileName}: ${transcript.length} chars`,
      );
      return transcript;
    } catch (error: any) {
      this.logger.error(
        `Whisper transcription failed for ${fileName}: ${error.message}`,
      );
      throw new Error(
        `Failed to transcribe audio "${fileName}": ${error.message}. The audio file may be corrupted, too long, or in an unsupported format.`,
      );
    } finally {
      // Clean up temp WAV file
      if (tempWavPath) {
        try {
          await fs.promises.unlink(tempWavPath);
        } catch {
          // Ignore cleanup errors
        }
      }
    }
  }

  /**
   * Check if a file is a supported media type
   */
  isSupported(filePath: string): boolean {
    const ext = path.extname(filePath).toLowerCase();
    return (
      this.imageExtensions.includes(ext) ||
      this.audioExtensions.includes(ext) ||
      this.textExtensions.includes(ext)
    );
  }

  /**
   * Get the embedding dimension for a media type
   */
  getEmbeddingDimension(): number {
    return this.TEXT_DIMENSION;
  }
}
