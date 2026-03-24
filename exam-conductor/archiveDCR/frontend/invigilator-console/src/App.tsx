import { RouterProvider } from "react-router-dom";
import { router } from "./router";

/**
 * Root application component.
 * Router is configured in router.tsx — this component simply mounts it.
 */
export default function App() {
  return <RouterProvider router={router} />;
}
