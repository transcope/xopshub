# xops工作空间
这里有公开的xops资料，包括调试完成的算例和数据集、阅读过的相关文献和资料。

## 模型集
这里添加算例。

## 数据集
这里添加数据集。

## 重要资料
这里添加相关文献和资料。

### 综合
* awesome系列收集了大量高质量资料，包括论文、数据集、工具包、领域知识和公司等。

	* aswsome-AIOps: <https://github.com/OpsPAI/awesome-AIOps>

 	* AIOps 手册: <https://github.com/chenryn/aiops-handbook>

### 算法
* [清华裴丹: AIOps落地的15条原则](https://bizseer.com/index.php?m=content&c=index&a=show&catid=26&id=63)

裴老师总结了模型落地相关经验，包括数据、人工智能、算法路线、数据质量、以及标准化。

* [调用链根因定位算法综述](https://dreamhomes.top/posts/202204281516/)

这篇博文总结十几篇调用链定位论文（最新研究论文汇总参见 <https://github.com/dreamhomes/RCAPapers>）。其中，Rank相关算法对数据要求较低，且没有系统结构要求。该类算法适合冷启动，适用于客户现场，尤其适合没有历史定位数据、没有图谱经验的客户。此外，该博客有提供算法的详细介绍（写作一般，启蒙较好）。

### 领域
* [图数据库在CMDB领域的应用](https://blog.csdn.net/joy0921/article/details/80132195?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165441247416781685349172%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165441247416781685349172&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-80132195-null-null.142^v11^pc_search_result_control_group,157^v13^control&utm_term=GRAPH+CMDB&spm=1018.2226.3001.4187)

这篇博文对CMDB领域的现状和问题分析比较全面。观点明确：图数据库用于CMDB应注重关联关系分析，而不是强调其数据库性能（因为数据规模没那么大）。此外，文章中有相关分析用例。

* 腾讯AIOps实践十分值得学习
	* [T4 级老专家：AIOps 在腾讯的探索和实践](https://cloud.tencent.com/developer/article/1362329)：属于非常接地气的工作总结，从规则到模型到演进，从单点项目开始的发展，落地遇到的困难和追求的目标，整个实践过程和项目规划的可行性路径值得借鉴（可能接近客户现场）。
	
	* [微众银行智能运维AIOps系列](https://www.zhihu.com/column/c_1256866843375628288)：该系列非常值得阅读和学习。但是，相较于上一篇，该系列不太适合落地，主要原因在于客户现场不太可能有微众那样的数据体系和组织方式。
 		* [AIOps的崛起与实践（一）](https://zhuanlan.zhihu.com/p/149095384) 
		* [智能化监控领域探索（二）](https://zhuanlan.zhihu.com/p/149250335)
		* [浅析智能异常检测：“慧识图”核心算法（三）](https://zhuanlan.zhihu.com/p/150316014)
		* [曝光交易路径（四）](https://zhuanlan.zhihu.com/p/154136946)
		* [浅析基于知识图谱的根因分析系统（五）](https://zhuanlan.zhihu.com/p/158059486)
		* [根因分析过程中的运维管理实践（六）](https://zhuanlan.zhihu.com/p/160767387)
		* [化繁为简：业务异常的根因定位方法概述（七）](https://zhuanlan.zhihu.com/p/162892617)
		* [事件指纹库：构建异常案例的“博物馆”（八）](https://zhuanlan.zhihu.com/p/166219885)
		* [基于交易树的根因告警定位方法（九）](https://zhuanlan.zhihu.com/p/178641464)
		* [浅析根因告警的系统分析法（十）](https://zhuanlan.zhihu.com/p/188634867)
		* [日志文本异常聚类及相似度检测（十一）](https://zhuanlan.zhihu.com/p/198918542)
		* [智能运维的四大挑战和应对之道（十二）](https://zhuanlan.zhihu.com/p/228201211)
		* [面向智能化运维的CMDB系统构建（十三）](https://zhuanlan.zhihu.com/p/250155094)
		* [人与技术相结合的异常管理实践（十四）](https://zhuanlan.zhihu.com/p/260758775)
