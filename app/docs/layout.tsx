import { source } from '@/lib/source';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { baseOptions } from '@/lib/layout.shared';
import { PublicAIChat } from '@/components/ai/public-ai-chat';
import { DocsSidebarFooter } from '@/components/docs-sidebar-footer';

export default function Layout({ children }: LayoutProps<'/docs'>) {
  return (
    <DocsLayout
      {...baseOptions()}
      tree={source.getPageTree()}
      tabs={false}
      sidebar={{
        collapsible: true,
        defaultOpenLevel: 2,
        footer: <DocsSidebarFooter />,
      }}
    >
      {children}
      <PublicAIChat />
    </DocsLayout>
  );
}
