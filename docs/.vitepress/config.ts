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
  description: '高性能 CUDA 图像处理库',
  lang: 'zh-CN',

  // 头部元信息
  head: [
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'Mini-OpenCV - GPU 图像处理库' }],
    ['meta', { property: 'og:description', content: '基于 CUDA 的高性能图像处理库，提供 GPU 加速算子。' }],
    ['meta', { property: 'og:image', content: '/mini-opencv/images/og-image.png' }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['link', { rel: 'icon', href: '/mini-opencv/favicon.ico' }],
  ],

  themeConfig: {
    outline: [2, 3],
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/AICL-Lab/mini-opencv' },
    ],
    footer: {
      message: '基于 MIT 许可证发布。',
      copyright: 'Copyright © 2024-present AICL-Lab',
    },
    editLink: {
      pattern: 'https://github.com/AICL-Lab/mini-opencv/edit/main/docs/:path',
      text: '在 GitHub 上编辑本页',
    },
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
