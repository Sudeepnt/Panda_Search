import {
  Controller,
  Delete,
  Get,
  Param,
  Post,
  HttpCode,
  HttpStatus,
} from '@nestjs/common';
import { IngestService } from './ingest.service';
import { DocumentListResponseDto } from './dto/ingest.dto';

@Controller('documents')
export class DocumentsController {
  constructor(private readonly ingestService: IngestService) {}

  @Get()
  @HttpCode(HttpStatus.OK)
  async listDocuments(): Promise<DocumentListResponseDto> {
    return this.ingestService.listDocuments();
  }

  @Post('deduplicate-images')
  @HttpCode(HttpStatus.OK)
  async deduplicateImages(): Promise<{ removed: number }> {
    const removed = await this.ingestService.deduplicateImageDocuments();
    return { removed };
  }

  @Post('prune-nonlocal')
  @HttpCode(HttpStatus.OK)
  async pruneNonLocal(): Promise<{ removed: number }> {
    const removed = await this.ingestService.pruneNonLocalDocuments();
    return { removed };
  }

  @Get('timings')
  @HttpCode(HttpStatus.OK)
  async indexTimings() {
    return this.ingestService.getIndexTimingSummary();
  }

  @Delete('all')
  @HttpCode(HttpStatus.OK)
  async deleteAllDocuments(): Promise<{ success: boolean; message: string }> {
    return this.ingestService.deleteAllDocuments();
  }

  @Delete(':id')
  @HttpCode(HttpStatus.OK)
  async deleteDocument(
    @Param('id') id: string,
  ): Promise<{ success: boolean; message: string }> {
    return this.ingestService.deleteDocument(id);
  }
}
