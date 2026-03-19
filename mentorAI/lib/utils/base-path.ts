const BASE_PATH = (process.env.NEXT_PUBLIC_BASE_PATH || '').replace(/\/$/, '');

export function withBasePath(path: string): string {
  if (!path) return BASE_PATH || '';
  if (
    path.startsWith('http://') ||
    path.startsWith('https://') ||
    path.startsWith('//') ||
    path.startsWith('data:') ||
    path.startsWith('blob:') ||
    path.startsWith('#')
  ) {
    return path;
  }

  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return BASE_PATH ? `${BASE_PATH}${normalizedPath}` : normalizedPath;
}

export { BASE_PATH };
