import { loader } from 'fumadocs-core/source';
import { docsRoute } from './shared';
import { defineDocs } from 'fumadocs-mdx/macro';
import { metaSchema, pageSchema } from 'fumadocs-core/source/schema';

const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    schema: pageSchema.extend({
      status: pageSchema.shape.title.optional(),
      lastVerified: pageSchema.shape.title.optional(),
      sourceCommit: pageSchema.shape.title.optional(),
      environment: pageSchema.shape.title.optional(),
      randomSeed: pageSchema.shape.title.optional(),
      reproduce: pageSchema.shape.title.optional(),
      reproducibility: pageSchema.shape.title.optional(),
    }),
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
  meta: {
    schema: metaSchema,
  },
});

// See https://fumadocs.dev/docs/headless/source-api for more info
export const source = loader({
  baseUrl: docsRoute,
  source: docs.toFumadocsSource(),
  plugins: [],
});
