import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { getToken, storeToken, isTokenExpired } from "@exampen/common-ts";
import { setTokenAccessor } from "@/api/student-api";
import App from "@/App";
import "@/index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
});

// Eagerly capture token from URL query param before React mounts,
// so the first API call in any component already has it in localStorage.
// The useAuth hook also does this reactively, but this ensures no race.
const params = new URLSearchParams(window.location.search);
const urlToken = params.get("token");
if (urlToken && !isTokenExpired(urlToken)) {
  storeToken(urlToken);
  const url = new URL(window.location.href);
  url.searchParams.delete("token");
  window.history.replaceState({}, "", url.toString());
}

// Wire up the auth token accessor so every API call carries the JWT.
// Reads from localStorage (key: exampen_token), populated by useAuth hook.
setTokenAccessor(() => getToken() ?? "");

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
