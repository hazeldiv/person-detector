import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Person Detector — CCTV Monitoring",
  description:
    "Real-time person detection for outdoor CCTV surveillance using HOG and background subtraction.",
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
