import { source } from '@/lib/source';
import { DocsLayout } from 'fumadocs-ui/layouts/notebook';
import { baseOptions } from '@/lib/layout.shared';
import { PublicAIChat } from '@/components/ai/public-ai-chat';

export default function Layout({ children }: LayoutProps<'/docs'>) {
  const { nav, ...base } = baseOptions();

  return (
    <DocsLayout
      {...base}
      tree={source.getPageTree()}
      tabMode="navbar"
      nav={{ ...nav, mode: 'top' }}
      sidebar={{ defaultOpenLevel: 1 }}
    >
      {children}
      <PublicAIChat />
    </DocsLayout>
  );
}
