# Model Comparison Report - 2026-08-03

> 本次测试针对 Top 文章，对比各主流大模型在论文总结（分类、前置科普与核心贡献）任务上的生成效果、耗时与成本。

## Paper 1: Directional Anisotropic Sensitivity Curves for Pulsar Timing Arrays

- **arXiv ID**: [2608.00250](https://arxiv.org/abs/2608.00250)
- **Categories**: Instrumentation and Methods for Astrophysics (astro-ph.IM), General Relativity and Quantum Cosmology (gr-qc)
- **Zotero Similarity Score**: `0.6086`

<details><summary><b>查看原摘要 (Abstract)</b></summary>

After two decades of observations, pulsar timing array collaborations have reported strong evidence for a stochastic gravitational wave background, most likely sourced by an inspiraling population of supermassive black hole binaries. Because that population is finite, anisotropy in the background is inevitable, producing hotspots on the sky that track the loudest binaries. Building on the Fisher formalism for anisotropic backgrounds, we recast the directional Fisher information as effective sensitivity curves on the sky, distinguishing the radiometer and full-Fisher estimators, along with a sky-weighted curve that reduces the anisotropic sky to a single spectrum and recovers the standard isotropic curve in the isotropic limit. We implement five sky-decomposition bases: pixel, spherical harmonic, square-root spherical harmonic, radiometer, and principal-map. We demonstrate the framework on an IPTA-like pulsar timing array: characterizing the angular response, analyzing the Fisher structure, recovering injected anisotropic hotspots, and forecasting future-array sensitivity. The framework is released as an extension to the sensitivity software package hasasia.

</details>

### 🤖 模型：`gemini-3.5-flash-lite`
- **耗时**: 6.05s | **Tokens**: Input 396, Output 374 (+ 1138 thinking tokens, Total Output Billed 1512) | **Cost**: $0.003899 (¥0.0263)

#### 🏷️ 主题分类 (`topic`)
Cosmology

#### 🔭 领域背景与前置科普 (`background_knowledge`)
脉冲星计时阵列（PTAs）通过高精度监测全天球毫秒脉冲星的脉冲到达时间，旨在探测低频引力波。随机引力波背景（GWB）通常被认为主要源于宇宙演化过程中超大质量黑洞双星系统的叠加辐射。由于这类源的数目有限，其在天球上的分布并非绝对均匀，必然会表现出空间各向异性（即形成引力波“热点”）。为了精确探测和约束这些空间分布特征，天文学家需要超越传统的各向异性假设，建立能够描述阵列对不同天区定向响应的灵敏度曲线与分析框架。

#### 🎯 核心工作与科学发现 (`contribution`)
该论文基于费雪信息矩阵形式主义（Fisher formalism），构建了脉冲星计时阵列的定向各向异性灵敏度曲线理论框架。主要贡献包括：1）区分并研究了辐射计（radiometer）与全费雪（full-Fisher）两种估计器，并引入了天区加权灵敏度曲线，从而将复杂的各向异性天区简化为单一谱，并在各向异性极限下平滑过渡到标准各向异性曲线；2）实现了五种天区分解基底（包括像素、球谐函数、平方根球谐函数、辐射计和主图）；3）利用类IPTA（国际脉冲星计时阵列）配置演示了该框架在表征角响应、分析费雪结构、恢复注入的各向异性热点以及预测未来阵列灵敏度方面的应用，并将相关工具作为 hasasia 软件包的扩展公开发布。

---

### 🤖 模型：`gemini-3.7-flash`
❌ **执行失败**: `Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.7-flash\nPlease retry in 23.906191498s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.7-flash'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '23s'}]}}]`

### 🤖 模型：`deepseek-v4-flash`
- **耗时**: 8.21s | **Tokens**: Input 489, Output 735 (+ 364 thinking tokens, Total Output Billed 735) | **Cost**: $0.000593 (¥0.0040)

#### 🏷️ 主题分类 (`topic`)
General Astrophysics

#### 🔭 领域背景与前置科普 (`background_knowledge`)
Pulsar timing arrays (PTAs) are a galactic-scale observatory that uses an array of precisely timed millisecond pulsars as detectors for gravitational waves. By monitoring the arrival times of radio pulses over years, PTAs are sensitive to gravitational waves with periods of months to decades, i.e., nanohertz frequencies. Recently, PTA collaborations have found strong evidence for a stochastic gravitational wave background (GWB), likely originating from the superposition of gravitational waves emitted by a cosmic population of inspiraling supermassive black hole binaries. This background is expected to be nearly isotropic, but because the binary population is finite, the background should actually be anisotropic, with 'hotspots' corresponding to particularly loud individual binaries. Characterizing this anisotropy is a key goal for future PTA analyses. The Fisher information formalism provides a statistical framework for estimating the uncertainty in measuring the angular distribution of the GWB, and can be used to define sensitivity curves that describe the ability of a PTA to detect anisotropic structures on the sky.

#### 🎯 核心工作与科学发现 (`contribution`)
This work recasts the directional Fisher information for anisotropic gravitational wave backgrounds as effective sensitivity curves on the sky. Three types of curves are distinguished: the radiometer and full-Fisher estimators, and a sky-weighted curve that reduces the anisotropic sky to a single spectrum, recovering the standard isotropic curve in the isotropic limit. The framework is implemented using five sky-decomposition bases: pixel, spherical harmonic, square-root spherical harmonic, radiometer, and principal-map. The method is demonstrated on an IPTA-like pulsar timing array, including characterization of the angular response, analysis of the Fisher structure, recovery of injected anisotropic hotspots, and forecasting future array sensitivity. The framework is released as an extension to the hasasia sensitivity software package.

---

### 🤖 模型：`deepseek-v4-pro`
- **耗时**: 22.99s | **Tokens**: Input 489, Output 1743 (+ 1426 thinking tokens, Total Output Billed 1743) | **Cost**: $0.003774 (¥0.0254)

#### 🏷️ 主题分类 (`topic`)
General Astrophysics

#### 🔭 领域背景与前置科普 (`background_knowledge`)
脉冲星计时阵列（PTA）利用毫秒脉冲星作为高精度时钟，通过长期监测其脉冲到达时间残差来探测频率在纳赫兹范围的引力波。该频段的主要预期源是星系并合过程中形成的超大质量黑洞双星（SMBHB）系统，它们旋近时辐射的引力波叠加形成随机背景。由于源数目有限且空间分布非均匀，背景可能呈现各向异性，即天空上某些方向信号更强。传统分析常假设各向同性，但研究各向异性能揭示局部宇宙结构和源分布信息。评估探测器对天空不同方向的灵敏度需要构建方向灵敏度曲线，常用Fisher信息矩阵来量化参数估计的不确定性，并可通过多种基函数（如像素、球谐函数等）对天空进行分解。

#### 🎯 核心工作与科学发现 (`contribution`)
本文在Fisher形式基础上，将方向Fisher信息重新表述为天空上的有效灵敏度曲线，区分了辐射计估计器和全Fisher估计器，并提出了天空加权曲线，该曲线在各向同性极限下还原为标准各向同性曲线。实现了五种天空分解基：像素、球谐函数、平方根球谐函数、辐射计和主图。在类似IPTA的脉冲星计时阵列上演示了该框架：表征了角度响应，分析了Fisher结构，成功恢复注入的各向异性热点，并预测了未来阵列的灵敏度。该框架作为hasasia灵敏度软件包的扩展发布。

---

### 🤖 模型：`qwen-plus`
- **耗时**: 10.13s | **Tokens**: Input 423, Output 446 | **Cost**: $0.000183 (¥0.0012)

#### 🏷️ 主题分类 (`topic`)
Cosmology

#### 🔭 领域背景与前置科普 (`background_knowledge`)
Pulsar Timing Arrays (PTAs) are galactic-scale gravitational wave detectors that monitor ultra-stable millisecond pulsars to search for nanohertz-frequency gravitational waves (GWs), primarily from inspiraling supermassive black hole binaries (SMBHBs) across cosmic time.

Unlike ground-based interferometers (e.g., LIGO), PTAs are sensitive to GWs with wavelengths comparable to the Solar System size (~light-years), making them ideal probes of low-frequency cosmological and astrophysical processes.

A stochastic gravitational wave background (SGWB) in PTAs is expected to be anisotropic—not uniform across the sky—because the underlying SMBHB population is finite, clustered in massive galaxies and galaxy groups, leading to spatially localized 'hotspots' of excess GW power.

Detecting and characterizing this anisotropy provides crucial astrophysical information: source demographics, binary formation channels, galaxy merger histories, and even potential hints of non-astrophysical origins (e.g., cosmic strings or early-universe phase transitions).

Standard PTA analyses assume isotropy (i.e., equal GW power in all directions), enabling inference of a single power-law spectrum—but this approximation breaks down as sensitivity improves and individual loud sources begin to emerge from the background.

#### 🎯 核心工作与科学发现 (`contribution`)
The paper introduces a framework for computing directional, anisotropic sensitivity curves for pulsar timing arrays, translating the Fisher information matrix for anisotropic stochastic backgrounds into intuitive sky-resolved sensitivity maps. It distinguishes between three estimator-specific curves: radiometer-weighted, full-Fisher, and sky-weighted (which recovers the standard isotropic sensitivity in the limit of perfect isotropy). The authors implement and compare five sky-decomposition bases (pixel, spherical harmonic, square-root spherical harmonic, radiometer, and principal-map), validate the method on an IPTA-like array, demonstrate recovery of injected anisotropic hotspots, characterize angular response and Fisher structure, and forecast future-array capabilities—all packaged as an open-source extension to the hasasia sensitivity software.

---

### 🤖 模型：`qwen3.6-27b`
- **耗时**: 59.36s | **Tokens**: Input 420, Output 3752 (+ 3279 thinking tokens, Total Output Billed 3752) | **Cost**: $0.010207 (¥0.0688)

#### 🏷️ 主题分类 (`topic`)
General Astrophysics

#### 🔭 领域背景与前置科普 (`background_knowledge`)
脉冲星计时阵列（PTA）利用毫秒脉冲星极高的自转稳定性，将其作为宇宙尺度的精密时钟来探测纳赫兹频段的引力波。当引力波穿过地球与脉冲星之间的时空时，会导致脉冲到达时间产生系统性延迟或提前，即计时残差。通过长期监测大量脉冲星的计时残差，PTA能够探测由超大质量黑洞双星并合等过程产生的引力波信号。近年来，国际多个PTA合作项目已初步确认了随机引力波背景的存在，其主要物理起源被认为是宇宙中大量演化中的超大质量黑洞双星系统。在早期数据分析中，为简化计算常假设该背景在全天分布是各向同性的。然而，由于可探测的强辐射源数量有限，实际背景必然呈现各向异性，天空上会出现与最亮双星位置对应的“热点”。因此，发展能够刻画PTA对全天不同方向引力波敏感度的理论工具与数据分析框架，已成为当前PTA天文学从“背景探测”迈向“源解析与天空成像”的关键研究范式。

#### 🎯 核心工作与科学发现 (`contribution`)
本文针对PTA探测各向异性随机引力波背景的需求，基于费雪信息矩阵理论，提出了一种将方向性费雪信息转化为全天有效灵敏度曲线的新框架。该工作明确区分了辐射计估计算法与完整费雪估计算法在灵敏度刻画上的差异，并引入一种全天加权灵敏度曲线，能够将复杂的各向异性天空响应压缩为单一谱线，且在物理极限下严格回归至传统的各向同性灵敏度曲线。研究系统实现并对比了五种天空分解基底（像素、球谐函数、平方根球谐函数、辐射计基底与主映射基底）的计算效能。通过在类国际脉冲星计时阵列（IPTA）模型上的数值演示，作者详细刻画了PTA的角响应特性，剖析了费雪矩阵的结构特征，成功验证了该方法对人工注入的各向异性热点的恢复能力，并对未来阵列的灵敏度进行了定量预测。该理论框架与代码已作为开源工具集成至hasasia软件包中，为PTA数据分析和源解析提供了标准化计算平台。

---
