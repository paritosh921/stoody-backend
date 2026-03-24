import { useState, useCallback } from "react";
import clsx from "clsx";

interface AnswerImageProps {
  src: string;
  alt: string;
}

/**
 * Zoomable image viewer. Click to toggle between fit-to-width and
 * zoomed-in view. In zoomed mode the image scrolls naturally.
 */
export default function AnswerImage({ src, alt }: AnswerImageProps) {
  const [zoomed, setZoomed] = useState(false);

  const toggle = useCallback(() => setZoomed((z) => !z), []);

  return (
    <div
      className={clsx(
        "relative overflow-auto rounded-md border border-gray-200 bg-gray-100",
        zoomed ? "max-h-[80vh] cursor-zoom-out" : "cursor-zoom-in",
      )}
      onClick={toggle}
    >
      <img
        src={src}
        alt={alt}
        className={clsx(
          "block transition-transform duration-200",
          zoomed ? "w-auto max-w-none scale-100" : "w-full",
        )}
      />
      <span className="absolute bottom-2 right-2 rounded bg-black/50 px-2 py-0.5 text-xs text-white">
        {zoomed ? "Click to fit" : "Click to zoom"}
      </span>
    </div>
  );
}
