import { source } from '@/lib/source';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { baseOptions } from '@/lib/layout.shared';
import { PublicAIChat } from '@/components/ai/public-ai-chat';
import { docsSidebarComponents } from '@/components/docs-sidebar';

export default function Layout({ children }: LayoutProps<'/docs'>) {
  return (
    <DocsLayout
      {...baseOptions()}
      tree={source.getPageTree()}
      tabs={false}
      sidebar={{
        collapsible: true,
        defaultOpenLevel: 2,
        components: docsSidebarComponents,
      }}
    >
      {children}
      <PublicAIChat />
    </DocsLayout>
  );
}
