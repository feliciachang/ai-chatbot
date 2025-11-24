import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    // Optional: don’t fail the build on ESLint errors either
    ignoreDuringBuilds: true,
  },
};

export default nextConfig;
