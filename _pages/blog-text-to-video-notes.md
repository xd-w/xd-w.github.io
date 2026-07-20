---
layout: archive
permalink: /blog/text-to-video-large-model-notes/
title: "文生视频大模型笔记"
author_profile: true
classes: t2v-note-page
---

<article class="blog-entry t2v-note-overview" markdown="1">
  <p class="academic-list__venue">Published: Jul 21, 2026</p>

这组笔记面向具备深度学习基础、希望系统进入文生视频（Text-to-Video, T2V）大模型研究与工程实践的读者。内容按三册组织：从概率生成建模、视频潜空间、扩散与 Flow Matching，到 Video DiT、MLLM Planner、主流系统、评测协议、长视频、后训练、安全与研究选题。

<div class="t2v-part-grid">
  <a class="t2v-part-card" href="/blog/text-to-video-large-model-notes/part-1/">
    <span>Part I</span>
    <strong>概率生成建模与时空扩散 Transformer</strong>
    <p>统一符号、Video VAE、DDPM、Flow Matching、Video DiT、MLLM Planner 与评测体系。</p>
  </a>
  <a class="t2v-part-card" href="/blog/text-to-video-large-model-notes/part-2/">
    <span>Part II</span>
    <strong>主流模型、系统工程与复现路线</strong>
    <p>模型谱系、Wan / Bernini、数据工程、训练系统、显存与成本估计、论文级复现。</p>
  </a>
  <a class="t2v-part-card" href="/blog/text-to-video-large-model-notes/part-3/">
    <span>Part III</span>
    <strong>评测、长视频、后训练与安全</strong>
    <p>评测证据链、长程一致性、世界模型、奖励建模、可信生成与研究路线图。</p>
  </a>
</div>

## 阅读建议

- 如果目标是读懂文生视频论文的方法部分，先读 Part I。
- 如果目标是复现开源模型或估算训练/推理成本，重点读 Part II。
- 如果目标是设计研究课题、评测协议或后训练方案，重点读 Part III。

## PDF 版本

- [上册 PDF](/files/text-to-video-notes/part1/text-to-video-textbook-part1.pdf)
- [中册 PDF](/files/text-to-video-notes/part2/text-to-video-textbook-part2.pdf)
- [下册 PDF](/files/text-to-video-notes/part3/text-to-video-textbook-part3.pdf)
</article>
