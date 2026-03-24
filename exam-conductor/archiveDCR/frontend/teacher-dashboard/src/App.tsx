// ---------------------------------------------------------------------------
// App — Root component. Providers are set up in main.tsx; this simply
// renders the router which includes the TeacherLayout wrapper.
// ---------------------------------------------------------------------------

import { AppRouter } from './router';

export function App() {
  return <AppRouter />;
}
