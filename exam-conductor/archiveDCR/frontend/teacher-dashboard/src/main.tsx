import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { storeToken, isTokenExpired } from '@exampen/common-ts';
import { AppRouter } from './router';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// Eagerly capture token from URL query param before React mounts,
// so the first API call in any component already has it in localStorage.
// The useAuth hook also does this reactively, but this ensures no race.
const params = new URLSearchParams(window.location.search);
const urlToken = params.get('token');
if (urlToken && !isTokenExpired(urlToken)) {
  storeToken(urlToken);
  const url = new URL(window.location.href);
  url.searchParams.delete('token');
  window.history.replaceState({}, '', url.toString());
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppRouter />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
