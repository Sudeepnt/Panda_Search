import { IsString, IsOptional, IsObject, IsEnum, IsBoolean } from 'class-validator';

export type MediaType = 'text' | 'image' | 'audio' | 'video';

export class IngestDocumentDto {
  @IsOptional()
  @IsString()
  content?: string; // Optional for media files

  @IsString()
  filePath: string;

  @IsOptional()
  @IsEnum(['text', 'image', 'audio', 'video'])
  mediaType?: MediaType; // Auto-detected if not provided

  @IsOptional()
  @IsObject()
  metadata?: Record<string, unknown>;

  /** Replace prior chunks/captions for this path before ingesting it again. */
  @IsOptional()
  @IsBoolean()
  forceReindex?: boolean;
}

export class IngestResponseDto {
  id: string;
  fileName: string;
  filePath: string;
  mediaType: MediaType;
  success: boolean;
  message: string;

  /** Present for images. These are generated locally during the one-time visual scan. */
  imageTitle?: string;
  imageDescription?: string;

  /** End-to-end backend time spent indexing this file, in milliseconds. */
  elapsedMs?: number;
}

export class DeleteDocumentDto {
  @IsString()
  id: string;
}

export class DocumentListResponseDto {
  documents: Array<{
    id: string;
    fileName: string;
    filePath: string;
    mediaType: MediaType;
    thumbnailPath?: string;
    metadata: Record<string, unknown>;
    /** All chunk IDs for this file (when chunked). Use these for search filtering. */
    chunkIds?: string[];
  }>;
  total: number;
}
