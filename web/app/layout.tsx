import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MatVision",
  description: "AI-powered wrestling film intelligence",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
