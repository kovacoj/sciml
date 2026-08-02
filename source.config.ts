import { remarkMarimo } from '@marimo-team/mdx-marimo/remark';
import { defineConfig } from 'fumadocs-mdx/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

export default defineConfig({
  mdxOptions: {
    remarkPlugins: (plugins) => [remarkMath, remarkMarimo, ...plugins],
    rehypePlugins: (plugins) => [rehypeKatex, ...plugins],
  },
});
