import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rusted Warfare Map Generator",
  description: "Static web wizard powered by Next.js + Rust WASM",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
