import type { NextConfig } from 'next';

const basePath = (process.env.NEXT_PUBLIC_BASE_PATH || '/mentor-ai').replace(/\/$/, '');

const nextConfig: NextConfig = {
  basePath: basePath || undefined,
  output: process.env.VERCEL ? undefined : 'standalone',
  transpilePackages: ['mathml2omml', 'pptxgenjs'],
  serverExternalPackages: [],
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },
  experimental: {
    proxyClientMaxBodySize: '200mb',
  },
};

export default nextConfig;
