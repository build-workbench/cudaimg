---
layout: home
hero:
  name: Mini-OpenCV
  text: ' '
  actions:
    - theme: brand
      text: 进入文档
      link: /zh/
---

<script setup>
import { onMounted } from 'vue'
import { useRouter } from 'vitepress'

onMounted(() => {
  const router = useRouter()
  router.go('/zh/')
})
</script>
