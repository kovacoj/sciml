import { Inter } from 'next/font/google';
import { Provider } from '@/components/provider';
import './global.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Scientific ML & Numerical PDEs — Research Notebook',
  description:
    'Research notes and reproducible computational experiments in scientific machine learning, numerical PDEs, CFD, and optimization.',
};

const inter = Inter({
  subsets: ['latin'],
});

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
