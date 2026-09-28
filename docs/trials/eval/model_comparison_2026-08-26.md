# Model Comparison Report - 2026-08-26

> 本次测试针对 Top 文章，对比各主流大模型在论文总结（分类、前置科普与核心贡献）任务上的生成效果、耗时与成本。

## Paper 1: An HI study of a large sample of ultra-diffuse galaxies

- **arXiv ID**: [2608.24225](https://arxiv.org/abs/2608.24225)
- **Categories**: astro-ph.GA
- **Zotero Similarity Score**: `0.6482`

<details><summary><b>查看原摘要 (Abstract)</b></summary>

By cross-matching the SMUDGes catalog with the FASHI and ALFALFA HI surveys, we construct an HI-detected sample of 112 ultra-diffuse galaxies (UDGs) and 48 low-surface-brightness (LSB) galaxies, providing HI-based redshifts for 76 galaxies for the first time. Using DESI DR1 redshifts, we assemble an HI-non-detected sample of 168 galaxies for stacking analysis, with detections in two stellar mass bins below 10^{8.5} M_sun. Combining optical, UV, and HI data, we investigate stellar masses, star formation rates, gas fractions, and kinematics. We present a systematic analysis of the HI mass--optical size relation for UDGs and LSBs, showing both populations follow a mass--optical size scaling similar to normal galaxies, suggesting an average HI surface density, under the assumption that the optical size traces the extent of the HI distribution. Their stellar mass--size relation indicates nearly constant stellar surface densities for galaxies with central surface brightness mu_{0,g}>~24mag arcsec^{-2}, independent of UDG or LSB classification. Many UDGs and LSB galaxies are systematically offset from the baryonic Tully--Fisher relation toward higher baryonic masses at a given velocity, consistent with trends found in UDGs with resolved HI kinematics. Both populations exhibit low star formation efficiencies and long gas depletion times, supporting multiple formation pathways for UDGs.

</details>

### 🤖 模型：`qwen3.8-flash`
- **耗时**: 118.43s | **Tokens**: Input 971, Output 11735 (+ 11099 thinking tokens, Total Output Billed 11735) | **Cost**: $0.000544 (¥0.0037)

#### 🏷️ 主题分类 (`topic`)
Milky Way and Local Universe

#### 🔭 领域背景与前置科普 (`background_knowledge`)
超弥散星系（Ultra-Diffuse Galaxies, UDGs）是一类具有大有效半径但表面亮度极低的天体，其核心表面亮度通常低于 26 mag arcsec⁻²，尺寸接近银河系却拥有类似矮星系的恒星质量。这类天体挑战了标准的星系形成模型，因为它们似乎缺乏典型的暗物质晕收缩历史或经历了剧烈的动力学演化。低表面亮度星系（LSBs）在性质上与 UDGs 有重叠，通常富含气体且恒星形成效率较低。中性氢（HI）作为主要的重子物质成分之一，是探测这些低光度、气体丰富星系的关键示踪剂。通过比较光学、紫外和 HI 数据，研究者可以重建恒星质量函数、气体性质及动力学关系（如 Tully-Fisher 关系），从而推断这些星系的形成机制是源于原初的气体丰富环境，还是潮汐剥离等后期动力学过程。

#### 🎯 核心工作与科学发现 (`contribution`)
- 构建了包含 112 个 HI 探测 UDGs 和 48 个 LSBs 的大样本，并首次为其中 76 个星系提供了基于 HI 的红移数据。利用 DESI DR1 红移，组装了一个包含 168 个未探测到 HI 的星系样本用于堆叠分析，发现低于 $10^{8.5}\text{ M}_\text{sun}$ 的两个恒星质量分箱中存在 HI 探测信号。
- **HI 质量与光学尺度关系**：系统分析了 HI 质量与光学有效半径的关系，发现 UDGs 和 LSBs 均遵循与正常星系相似的质量-尺度标度律。这表明在假设光学尺度能够示踪 HI 分布范围的前提下，这两类星系具有平均一致的 HI 表面密度。
- **恒星质量-尺度关系的均匀性**：对于核心表面亮度 $\text{mu}_{0,g} \textgreater \textasciitilde 24\text{ mag arcsec}^{-2}$ 的星系，其恒星质量-尺度关系显示恒星表面密度近乎恒定，且这一特征与 UDG 或 LSB 的分类标签无关，暗示表面亮度阈值可能是决定这些结构性质的关键因素。
- **偏离重子 Tully-Fisher 关系**：许多 UDGs 和 LSBs 在重子 Tully-Fisher 关系上表现出系统性的偏移，即在给定的旋转速度下具有**更高的重子质量**。这一趋势与之前对具有分解 HI 运动学的 UDGs 的研究结果一致，可能反映了非平衡态或特殊的暗物质-重子分布比例。
- **低恒星形成效率与多路径形成**：两类星系均表现出极低的恒星形成效率和长的气体耗尽时间（Gas Depletion Times），支持 UDGs 并非由单一机制形成，而是存在**多种演化路径**（如原初低密度环境或后期剥离）。

---

## Paper 2: First application of weak lensing peak steepness statistics to HSC Y1 data: effectively probing halo density profiles

- **arXiv ID**: [2608.24591](https://arxiv.org/abs/2608.24591)
- **Categories**: astro-ph.CO
- **Zotero Similarity Score**: `0.5927`

<details><summary><b>查看原摘要 (Abstract)</b></summary>

As a new probe, the weak lensing (WL) peak steepness statistics is sensitive to the density profile of halos that encodes important information of baryonic feedback and dark matter properties, leading to a promising means to statistically constrain these effects using WL data. In this article, we present its first application to HSC Y1 data to demonstrate the great potential of this new statistics. Within the phenomenological framework of HMcode2016 that attributes the baryonic feedback solely to the reduction of the halo concentration parameter and focusing on high peaks originated dominantly from massive clusters, our analyses by combining WL peak height and steepness statistics resulted in $S_8=0.76^{+0.08}_{-0.07}$ with the maximum-a-posteriori (MAP) of $0.79$ and low concentrations. Taking the form of the concentration-mass relation as $c(M,z)=A(1+z_{\rm f})/(1+z)$ with $z_{\rm f}$ being the formation redshift of halos with mass $M$ at redshift $z$, we obtain $A=1.93^{+1.33}_{-1.16}$ (MAP=$1.70$) in comparison with $A=3.34^{+1.52}_{-1.74}$ (MAP=3.31) from dark matter only simulated mocks. The result tends to support phenomenologically strong baryonic feedback effects at cluster scales.

</details>

### 🤖 模型：`qwen3.8-flash`
- **耗时**: 85.62s | **Tokens**: Input 982, Output 8489 (+ 7994 thinking tokens, Total Output Billed 8489) | **Cost**: $0.000400 (¥0.0027)

#### 🏷️ 主题分类 (`topic`)
Large-Scale Structure

#### 🔭 领域背景与前置科普 (`background_knowledge`)
在宇宙学研究中，弱引力透镜（Weak Lensing, WL）是探测暗物质分布和限制宇宙学参数（如结构增长幅度 $S_8$）的最有力工具之一。传统方法主要依赖角功率谱或剪切相关函数，而“峰统计”（Peak Statistics）则关注剪切场中局部极大值的分布特性。其中，“峰陡峭度”（Peak Steepness）是一个较新的探针，它描述了峰在空间上的衰减快慢，对引力势阱的密度剖面形状（尤其是晕中心区域的聚集程度）极为敏感。由于重子物理过程（如超新星反馈、活动星系核喷流）和暗物质性质（如自相互作用）都会显著改变暗物质晕的密度剖面（特别是集中度 $c$），因此结合峰高度与陡峭度统计，有望打破参数简并，更精确地约束重子反馈效应及暗物质属性。

#### 🎯 核心工作与科学发现 (`contribution`)
- 首次将弱透镜峰陡峭度统计应用于 Subaru HSC Y1 数据，展示了该新统计量在探测晕密度剖面方面的巨大潜力。
- 基于 HMcode2016 唯象框架，假设重子反馈效应主要体现为暗物质晕集中度参数的降低，并聚焦于主要来自大质量星系团的高信号峰进行分析。
- 联合弱透镜峰高度与陡峭度统计，测得结构增长参数 **$S_8=0.76^{+0.08}_{-0.07}$**（最大后验概率 MAP 为 0.79），并发现晕具有较低的集中度。
- 采用集中度-质量关系 $c(M,z)=A(1+z_{\rm f})/(1+z)$（其中 $z_{\rm f}$ 为形成红移），拟合得到系数 **$A=1.93^{+1.33}_{-1.16}$**（MAP=1.70），显著低于纯暗物质模拟预测值 $A=3.34^{+1.52}_{-1.74}$（MAP=3.31）。
- 这一结果倾向于支持在星系团尺度上存在**强烈的重子反馈效应**，导致晕集中度显著降低。

---

## Paper 3: Exact Equivalence of the Observed Redshift and the Pulsar Timing Modulation in the Infinitesimal-Pulse Limit

- **arXiv ID**: [2608.23684](https://arxiv.org/abs/2608.23684)
- **Categories**: astro-ph.CO
- **Zotero Similarity Score**: `0.5799`

<details><summary><b>查看原摘要 (Abstract)</b></summary>

The pulsar timing arrays (PTA) collect the times of arrival of radio signals from the millisecond pulsars, and gravitational waves can modulate their arrival times. Although the PTA observable involves successive radio pulses propagating along different light paths, its standard theoretical description at linear order in perturbations is equivalent to the Sachs-Wolfe formula for the observed redshift, which describes the fractional change in photon frequency along a single geodesic. While this equivalence is well known at linear order, its validity beyond first order has not been systematically addressed in the PTA literature. Here we show that, in the limit of vanishing proper-time separation between successive emission events, the timing modulation is exactly equal to the observed redshift, without expanding the spacetime geometry. For a finite emission interval, we derive the exact relation between the timing modulation and the observed redshift, and show that their difference is controlled by the ratio between the emission interval and the characteristic timescale over which the observed redshift varies.

</details>

### 🤖 模型：`qwen3.8-flash`
- **耗时**: 41.59s | **Tokens**: Input 870, Output 3898 (+ 3544 thinking tokens, Total Output Billed 3898) | **Cost**: $0.000193 (¥0.0013)

#### 🏷️ 主题分类 (`topic`)
Gravitational Lensing

#### 🔭 领域背景与前置科普 (`background_knowledge`)
脉冲星计时阵列（PTA）通过监测毫秒脉冲星射电信号到达地球的时间（TOA）来探测纳赫兹频段的引力波背景。当引力波经过时，它会扰动时空度规，导致脉冲信号在传播路径上的光行时产生微小的变化。在传统的线性微扰理论框架下，PTA 的观测物理量（计时残差）在数学形式上与宇宙学中的 Sachs-Wolfe 效应完全等价。Sachs-Wolfe 效应描述了光子在沿单条零测地线传播过程中，因引力势阱的时间变化而产生的频率偏移（即红移）。尽管这一线性等价性在 PTA 数据分析中被广泛采用作为近似，但在高阶引力波效应或强场情形下，这种基于微扰展开的对应关系是否依然严格成立，以及有限时间间隔下的修正项如何影响精密检验，是检验广义相对论强场效应和系统误差分析的重要理论问题。

#### 🎯 核心工作与科学发现 (`contribution`)
- 本文证明了在连续发射事件的固有时分离趋于零的极限下，PTA 的计时调制与观测红移之间存在 **精确的数学等价性**，这一结论不依赖于时空几何的线性微扰展开。
- 针对有限发射时间间隔的一般情况，研究推导了计时调制与观测红移之间的 **精确解析关系**。
- 理论分析指出，在有限时间间隔下，两者之间的差异由 **发射间隔与观测红移变化特征时间尺度的比值** 决定，为理解 PTA 数据的高阶系统误差提供了明确的物理量度。

---
