import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import llmstxt from 'vitepress-plugin-llms'

const rawBase = process.env.VITEPRESS_BASE
const base = rawBase
  ? rawBase.startsWith('/')
    ? rawBase.endsWith('/') ? rawBase : `${rawBase}/`
    : `/${rawBase}/`
  : '/mini-opencv/'

export default withMermaid(defineConfig({
  base,
  title: 'Mini-OpenCV',
  description: 'High-Performance CUDA Image Processing Library',

  // 头部元信息
  head: [
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'Mini-OpenCV - GPU Image Processing Library' }],
    ['meta', { property: 'og:description', content: 'High-performance CUDA-based image processing library with GPU-accelerated operators.' }],
    ['meta', { property: 'og:image', content: '/mini-opencv/images/og-image.png' }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['link', { rel: 'icon', href: '/mini-opencv/favicon.ico' }],
  ],

  locales: {
    root: {
      label: 'Language',
      lang: 'en',
      link: '/',
    },
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      title: 'Mini-OpenCV',
      description: 'High-Performance CUDA Image Processing Library',
      themeConfig: {
        nav: [
          { text: 'Docs', link: '/en/setup/quickstart', activeMatch: '/en/' },
          { text: 'Architecture', link: '/en/architecture/overview', activeMatch: '/en/architecture/' },
          { text: 'API', link: '/en/api/', activeMatch: '/en/api/' },
          { text: 'Benchmarks', link: '/en/benchmarks/', activeMatch: '/en/benchmarks/' },
          { text: 'References', link: '/en/references/', activeMatch: '/en/references/' },
        ],
        sidebar: {
          '/en/': [
            {
              text: 'Technical Whitepaper',
              collapsed: false,
              items: [
                { text: 'Overview', link: '/en/whitepaper/overview' },
              ],
            },
            {
              text: 'Getting Started',
              collapsed: false,
              items: [
                { text: 'Quick Start', link: '/en/setup/quickstart' },
                { text: 'Installation', link: '/en/setup/installation' },
                { text: 'Configuration', link: '/en/setup/configuration' },
              ],
            },
            {
              text: 'Architecture',
              collapsed: false,
              items: [
                { text: 'Overview', link: '/en/architecture/overview' },
                { text: 'Memory Model', link: '/en/architecture/memory-model' },
                { text: 'CUDA Streams', link: '/en/architecture/cuda-streams' },
                { text: 'Design Decisions', link: '/en/architecture/design-decisions' },
              ],
            },
            {
              text: 'API Reference',
              collapsed: false,
              items: [
                { text: 'Overview', link: '/en/api/' },
                { text: 'ImageProcessor', link: '/en/api/core/image-processor' },
                { text: 'GpuImage', link: '/en/api/core/gpu-image' },
                { text: 'DeviceBuffer', link: '/en/api/core/device-buffer' },
                { text: 'Convolution Ops', link: '/en/api/operators/convolution' },
                { text: 'Morphology Ops', link: '/en/api/operators/morphology' },
                { text: 'Geometric Ops', link: '/en/api/operators/geometric' },
                { text: 'Filters', link: '/en/api/operators/filters' },
              ],
            },
            {
              text: 'Benchmarks',
              collapsed: true,
              items: [
                { text: 'Overview', link: '/en/benchmarks/' },
                { text: 'Convolution', link: '/en/benchmarks/convolution' },
                { text: 'Methodology', link: '/en/benchmarks/methodology' },
              ],
            },
            {
              text: 'Tutorials',
              collapsed: true,
              items: [
                { text: 'Examples', link: '/en/tutorials/examples' },
                { text: 'Advanced', link: '/en/tutorials/advanced' },
                { text: 'FAQ', link: '/en/tutorials/faq' },
              ],
            },
            {
              text: 'References',
              collapsed: true,
              items: [
                { text: 'Papers & Projects', link: '/en/references/' },
              ],
            },
          ],
        },
      },
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      title: 'Mini-OpenCV',
      description: '高性能 CUDA 图像处理库',
      themeConfig: {
        nav: [
          { text: '文档', link: '/zh/setup/quickstart', activeMatch: '/zh/' },
          { text: '架构', link: '/zh/architecture/overview', activeMatch: '/zh/architecture/' },
          { text: 'API', link: '/zh/api/', activeMatch: '/zh/api/' },
          { text: '性能', link: '/zh/benchmarks/', activeMatch: '/zh/benchmarks/' },
          { text: '引用', link: '/zh/references/', activeMatch: '/zh/references/' },
        ],
        sidebar: {
          '/zh/': [
            {
              text: '技术白皮书',
              collapsed: false,
              items: [
                { text: '技术概述', link: '/zh/whitepaper/overview' },
              ],
            },
            {
              text: '快速开始',
              collapsed: false,
              items: [
                { text: '快速入门', link: '/zh/setup/quickstart' },
                { text: '安装指南', link: '/zh/setup/installation' },
                { text: '配置说明', link: '/zh/setup/configuration' },
              ],
            },
            {
              text: '架构设计',
              collapsed: false,
              items: [
                { text: '架构概览', link: '/zh/architecture/overview' },
                { text: '内存模型', link: '/zh/architecture/memory-model' },
                { text: 'CUDA 流并发', link: '/zh/architecture/cuda-streams' },
                { text: '设计决策', link: '/zh/architecture/design-decisions' },
              ],
            },
            {
              text: 'API 参考',
              collapsed: false,
              items: [
                { text: '概览', link: '/zh/api/' },
                { text: 'ImageProcessor', link: '/zh/api/core/image-processor' },
                { text: 'GpuImage', link: '/zh/api/core/gpu-image' },
                { text: 'DeviceBuffer', link: '/zh/api/core/device-buffer' },
                { text: '卷积操作', link: '/zh/api/operators/convolution' },
                { text: '形态学操作', link: '/zh/api/operators/morphology' },
                { text: '几何变换', link: '/zh/api/operators/geometric' },
                { text: '滤波器', link: '/zh/api/operators/filters' },
              ],
            },
            {
              text: '性能基准',
              collapsed: true,
              items: [
                { text: '概览', link: '/zh/benchmarks/' },
                { text: '卷积性能', link: '/zh/benchmarks/convolution' },
                { text: '测试方法', link: '/zh/benchmarks/methodology' },
              ],
            },
            {
              text: '教程',
              collapsed: true,
              items: [
                { text: '示例代码', link: '/zh/tutorials/examples' },
                { text: '高级主题', link: '/zh/tutorials/advanced' },
                { text: '常见问题', link: '/zh/tutorials/faq' },
              ],
            },
            {
              text: '学术引用',
              collapsed: true,
              items: [
                { text: '论文与项目', link: '/zh/references/' },
              ],
            },
          ],
        },
      },
    },
  },

  themeConfig: {
    outline: [2, 3],
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/AICL-Lab/mini-opencv' },
    ],
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present AICL-Lab',
    },
    editLink: {
      pattern: 'https://github.com/AICL-Lab/mini-opencv/edit/main/docs/:path',
      text: 'Edit this page on GitHub',
    },
  },

  vite: {
    plugins: [llmstxt()],
  },

  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark',
    },
  },

  mermaid: {
    theme: 'base',
    themeVariables: {
      primaryColor: '#76B900',
      primaryTextColor: '#24292f',
      primaryBorderColor: '#76B900',
      lineColor: '#57606a',
      secondaryColor: '#f6f8fa',
      tertiaryColor: '#d0d7de',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    },
    flowchart: {
      curve: 'basis',
      padding: 15,
    },
    sequence: {
      actorMargin: 50,
      boxMargin: 10,
    },
  },
}))
