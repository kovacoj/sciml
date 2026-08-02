import { FullSearchTrigger } from 'fumadocs-ui/layouts/shared/slots/search-trigger';
import { SidebarCollapseTrigger } from 'fumadocs-ui/layouts/notebook/slots/sidebar';

export function DocsTocHeader() {
  return (
    <div className="flex items-center gap-2">
      <FullSearchTrigger className="min-w-0 flex-1" />
      <SidebarCollapseTrigger />
    </div>
  );
}
