import { remarkMarimo } from '@marimo-team/mdx-marimo/remark';
import { defineConfig } from 'fumadocs-mdx/config';

export default defineConfig({
  mdxOptions: {
    remarkPlugins: (plugins) => [remarkMarimo, ...plugins],
  },
});
