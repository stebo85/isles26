import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "@xyflow/react/dist/style.css";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export async function generateMetadata(): Promise<Metadata> {
  let metadataBase = new URL("http://localhost:3000");
  if (process.env.SITE_URL) {
    try {
      const configured = new URL(process.env.SITE_URL);
      if (configured.protocol === "http:" || configured.protocol === "https:") {
        metadataBase = configured;
      }
    } catch {
      // A malformed deployment setting must not make page rendering fail.
    }
  }
  const title = "RepoTrace · Repository provenance explorer";
  const description =
    "Trace a repository from evidence through decisions to its current pipeline.";

  return {
    metadataBase,
    title,
    description,
    openGraph: {
      title,
      description,
      type: "website",
      images: [{ url: "/og.png", width: 1731, height: 909, alt: "RepoTrace decision provenance graph" }],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: ["/og.png"],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
