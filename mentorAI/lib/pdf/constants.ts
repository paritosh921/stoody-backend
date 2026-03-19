/**
 * PDF Provider Constants
 * Separated from pdf-providers.ts to avoid importing sharp in client components
 */

import type { PDFProviderId, PDFProviderConfig } from './types';
import { withBasePath } from '@/lib/utils/base-path';

/**
 * PDF Provider Registry
 */
export const PDF_PROVIDERS: Record<PDFProviderId, PDFProviderConfig> = {
  unpdf: {
    id: 'unpdf',
    name: 'unpdf',
    requiresApiKey: false,
    icon: withBasePath('/logos/unpdf.svg'),
    features: ['text', 'images', 'metadata'],
  },

  mineru: {
    id: 'mineru',
    name: 'MinerU',
    requiresApiKey: false,
    icon: withBasePath('/logos/mineru.png'),
    features: ['text', 'images', 'tables', 'formulas', 'layout-analysis'],
  },
};

/**
 * Get all available PDF providers
 */
export function getAllPDFProviders(): PDFProviderConfig[] {
  return Object.values(PDF_PROVIDERS);
}

/**
 * Get PDF provider by ID
 */
export function getPDFProvider(providerId: PDFProviderId): PDFProviderConfig | undefined {
  return PDF_PROVIDERS[providerId];
}
