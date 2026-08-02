'use client';

import type * as PageTree from 'fumadocs-core/page-tree';
import type { ComponentProps, ReactNode } from 'react';
import {
  SidebarFolder,
  SidebarFolderContent,
  SidebarFolderLink,
  SidebarFolderTrigger,
  SidebarItem,
  SidebarSeparator,
  useFolderDepth,
} from 'fumadocs-ui/components/sidebar/base';
import type { SidebarPageTreeComponents } from 'fumadocs-ui/components/sidebar/page-tree';

function FolderContent({ children, ...props }: ComponentProps<typeof SidebarFolderContent>) {
  const depth = useFolderDepth();

  return (
    <SidebarFolderContent
      {...props}
      className={depth >= 1 ? 'relative before:absolute before:inset-y-1 before:start-2.5 before:w-px before:bg-fd-border' : undefined}
    >
      {children}
    </SidebarFolderContent>
  );
}

export const docsSidebarComponents: Partial<SidebarPageTreeComponents> = {
  Item: ({ item }: { item: PageTree.Item }) => (
    <SidebarItem
      href={item.url}
      className="rounded-md py-1.5 text-fd-muted-foreground data-[active=true]:font-medium data-[active=true]:text-cyan-600"
    >
      {item.icon}
      {item.name}
    </SidebarItem>
  ),
  Folder: ({ item, children }: { item: PageTree.Folder; children: ReactNode }) => (
    <SidebarFolder>
      {item.index ? (
        <SidebarFolderLink href={item.index.url}>{item.name}</SidebarFolderLink>
      ) : (
        <SidebarFolderTrigger className="w-full font-semibold text-fd-muted-foreground hover:text-fd-foreground">
          {item.name}
        </SidebarFolderTrigger>
      )}
      <FolderContent>{children}</FolderContent>
    </SidebarFolder>
  ),
  Separator: ({ item }: { item: PageTree.Separator }) => (
    <SidebarSeparator>{item.name}</SidebarSeparator>
  ),
};
