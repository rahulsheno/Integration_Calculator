import { useEffect, useRef } from 'react';
import katex from 'katex';

interface Props {
  latex: string;
  displayMode?: boolean;
  className?: string;
}

export default function MathRenderer({ latex, displayMode = false, className = '' }: Props) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (ref.current && latex) {
      try {
        katex.render(latex, ref.current, {
          displayMode,
          throwOnError: false,
          // This renders LaTeX that can originate from user input or OCR/
          // vision-LLM output, none of which is trusted. `trust: true` would
          // enable \href, \includegraphics, \url etc. on that untrusted
          // content, which is a real XSS-adjacent risk - keep it disabled.
          trust: false,
        });
      } catch {
        ref.current.textContent = latex;
      }
    }
  }, [latex, displayMode]);

  return <span ref={ref} className={className} />;
}
