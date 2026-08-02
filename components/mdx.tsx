import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import { MarimoNotebook } from '@/components/marimo-notebook';

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    MarimoNotebook,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
