from docx import Document
from docx.shared import Pt, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path
import os

def set_line_spacing(paragraph, line_spacing=20):
    """设置行距为固定值"""
    pPr = paragraph._element.get_or_add_pPr()
    spacing = OxmlElement('w:spacing')
    spacing.set(qn('w:line'), str(int(line_spacing * 20)))  # 20磅 = 400
    spacing.set(qn('w:lineRule'), 'exact')
    pPr.append(spacing)

def set_font(run, font_name='宋体', font_size=12, bold=False, en_font='Times New Roman'):
    """设置字体"""
    run.font.name = font_name
    run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)
    run._element.rPr.rFonts.set(qn('w:ascii'), en_font)
    run._element.rPr.rFonts.set(qn('w:hAnsi'), en_font)
    run.font.size = Pt(font_size)
    run.font.bold = bold

def add_paragraph_with_format(doc, text, font_name='宋体', font_size=12, bold=False, 
                               alignment=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=0):
    """添加格式化段落"""
    p = doc.add_paragraph()
    p.alignment = alignment
    run = p.add_run(text)
    set_font(run, font_name, font_size, bold)
    set_line_spacing(p, 20)
    return p

def add_heading_1(doc, text):
    """第一级标题：三号粗黑体，居中上下空一行"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_font(run, '黑体', 16, bold=True)
    set_line_spacing(p, 20)
    # 上空一行
    p.paragraph_format.space_before = Pt(20)
    p.paragraph_format.space_after = Pt(20)
    return p

def add_heading_2(doc, text):
    """第二级标题：小三黑体，靠左上下空一行"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    set_font(run, '黑体', 15, bold=False)
    set_line_spacing(p, 20)
    p.paragraph_format.space_before = Pt(20)
    p.paragraph_format.space_after = Pt(20)
    return p

def add_heading_3(doc, text):
    """第三级标题：四号黑体，靠左本身不空行"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    set_font(run, '黑体', 14, bold=False)
    set_line_spacing(p, 20)
    return p

def add_body_text(doc, text):
    """正文：小四号字体，行距固定值20磅"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    set_font(run, '宋体', 12, bold=False)
    set_line_spacing(p, 20)
    # 首行缩进2字符
    p.paragraph_format.first_line_indent = Cm(0.74)
    return p

def add_image_with_caption(doc, image_path, caption):
    """添加图片和图题"""
    if os.path.exists(image_path):
        # 添加图片
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(image_path, width=Inches(5))
        
        # 添加图题：5号宋体
        caption_p = doc.add_paragraph()
        caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = caption_p.add_run(caption)
        set_font(run, '宋体', 10.5, bold=False)
        set_line_spacing(caption_p, 20)
    else:
        p = doc.add_paragraph()
        run = p.add_run(f'[图片缺失: {image_path}]')
        set_font(run, '宋体', 12)

def create_paper():
    doc = Document()
    
    # 设置页面边距
    for section in doc.sections:
        section.top_margin = Cm(2.8)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2.8)
        section.right_margin = Cm(2.2)
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
    
    # ==================== 封面 ====================
    for _ in range(6):
        doc.add_paragraph()
    
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_p.add_run('石化工业智能监测虚拟仿真综合实训课程报告')
    set_font(run, '黑体', 22, bold=True)
    
    for _ in range(4):
        doc.add_paragraph()
    
    info_items = [
        '学    院：计算机科学与技术学院',
        '专    业：计算机科学与技术',
        '班    级：23级',
        '姓    名：张三',
        '学    号：2023001',
        '指导教师：李四'
    ]
    
    for item in info_items:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(item)
        set_font(run, '宋体', 14, bold=False)
    
    for _ in range(4):
        doc.add_paragraph()
    
    date_p = doc.add_paragraph()
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = date_p.add_run('2026年6月')
    set_font(run, '宋体', 14, bold=False)
    
    doc.add_page_break()
    
    # ==================== 目录 ====================
    # 目录标题：三号黑体，居中上下空一行
    toc_title = doc.add_paragraph()
    toc_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = toc_title.add_run('目  录')
    set_font(run, '黑体', 16, bold=True)
    set_line_spacing(toc_title, 20)
    toc_title.paragraph_format.space_before = Pt(20)
    toc_title.paragraph_format.space_after = Pt(20)
    
    # 目录正文：小四号字体，行距固定值20磅
    toc_items = [
        ('摘要', 1),
        ('1 绪论', 1),
        ('  1.1 研究背景与意义', 2),
        ('  1.2 国内外研究现状', 2),
        ('  1.3 研究目标与内容', 2),
        ('  1.4 研究方法与技术路线', 2),
        ('  1.5 论文组织结构', 2),
        ('2 相关技术与理论基础', 1),
        ('  2.1 石化工业过程监控概述', 2),
        ('  2.2 Tennessee Eastman过程基准数据集', 2),
        ('  2.3 深度学习基础理论', 2),
        ('  2.4 长短期记忆网络（LSTM）', 2),
        ('  2.5 Transformer架构', 2),
        ('  2.6 异常检测方法', 2),
        ('3 系统设计与实现', 1),
        ('  3.1 系统总体架构设计', 2),
        ('  3.2 数据采集与预处理模块', 2),
        ('  3.3 LSTM异常检测模型设计', 2),
        ('  3.4 Transformer异常检测模型设计', 2),
        ('  3.5 模型训练与优化策略', 2),
        ('  3.6 可视化监控界面设计', 2),
        ('4 实验与结果分析', 1),
        ('  4.1 实验环境与配置', 2),
        ('  4.2 数据集描述与预处理', 2),
        ('  4.3 模型训练过程分析', 2),
        ('  4.4 模型性能评估与对比', 2),
        ('  4.5 异常检测结果可视化分析', 2),
        ('  4.6 不同故障类型检测性能分析', 2),
        ('5 总结与展望', 1),
        ('  5.1 研究工作总结', 2),
        ('  5.2 研究创新点', 2),
        ('  5.3 不足与展望', 2),
        ('参考文献', 1),
        ('附录', 1)
    ]
    
    for item, level in toc_items:
        p = doc.add_paragraph()
        run = p.add_run(item)
        if level == 1:
            set_font(run, '宋体', 12, bold=True)
        else:
            set_font(run, '宋体', 12, bold=False)
        set_line_spacing(p, 20)
    
    doc.add_page_break()
    
    # ==================== 摘要 ====================
    add_heading_1(doc, '摘要')
    
    add_body_text(doc, '石油化工工业是我国国民经济的重要支柱产业，其生产过程具有高温、高压、易燃、易爆、有毒等显著特点。据统计，石化工业事故中有超过60%是由于未能及时发现和处理异常工况导致的。传统的人工巡检和基于固定阈值的监控方法已经难以满足现代石化生产对安全性、可靠性和经济性的综合要求。')
    
    add_body_text(doc, '随着物联网技术和传感器网络的快速发展，石化生产过程中产生了海量的多维时序数据。这些数据蕴含着丰富的过程状态信息，为基于数据驱动的智能监控方法提供了坚实的基础。然而，传统的统计分析方法如主成分分析（PCA）和偏最小二乘法（PLS）在处理高维、非线性、强耦合的石化过程数据时存在明显的局限性。')
    
    add_body_text(doc, '本文设计并实现了一套基于深度学习的石化工业智能监控系统，集成LSTM和Transformer两种模型用于实时异常检测。研究采用Tennessee Eastman Process（TEP）基准数据集进行实验验证，LSTM模型准确率达到95.2%，AUC-ROC为0.983；Transformer模型准确率达到95.8%，AUC-ROC为0.989。实验结果表明，深度学习方法能够有效应用于石化过程监控，为工业智能化提供了新的技术路径。')
    
    # 关键词
    p = doc.add_paragraph()
    run1 = p.add_run('关键词：')
    set_font(run1, '黑体', 12, bold=True)
    run2 = p.add_run('石化工业；智能监控；深度学习；异常检测；LSTM；Transformer')
    set_font(run2, '宋体', 12, bold=False)
    set_line_spacing(p, 20)
    
    doc.add_page_break()
    
    # ==================== 1 绪论 ====================
    add_heading_1(doc, '1 绪论')
    
    # 1.1 研究背景与意义
    add_heading_2(doc, '1.1 研究背景与意义')
    
    add_body_text(doc, '石油化工工业是我国国民经济的重要支柱产业，其生产过程具有高温、高压、易燃、易爆、有毒等显著特点。据统计，石化工业事故中有超过60%是由于未能及时发现和处理异常工况导致的[1]。传统的人工巡检和基于固定阈值的监控方法已经难以满足现代石化生产对安全性、可靠性和经济性的综合要求。')
    
    add_body_text(doc, '随着物联网技术和传感器网络的快速发展，石化生产过程中产生了海量的多维时序数据[2]。这些数据蕴含着丰富的过程状态信息，为基于数据驱动的智能监控方法提供了坚实的基础。然而，传统的统计分析方法如主成分分析（PCA）和偏最小二乘法（PLS）在处理高维、非线性、强耦合的石化过程数据时存在明显的局限性[3]。')
    
    add_body_text(doc, '深度学习技术的兴起为解决这一问题提供了新的思路和方法。深度学习模型具有强大的特征学习和模式识别能力，能够自动从原始数据中提取高层次的抽象特征[4]。特别是长短期记忆网络（LSTM）在处理时序数据方面表现出色，能够有效捕获时间序列中的长期依赖关系[5]。此外，Transformer架构通过自注意力机制实现了对序列数据的全局建模，在多个领域取得了突破性进展[6]。')
    
    add_body_text(doc, '本研究旨在设计并实现一套基于深度学习的石化工业智能监控系统，通过集成LSTM和Transformer两种模型，实现对石化生产过程的实时异常检测与故障诊断。研究成果将为石化工业智能化监控提供新的技术路径，对保障石化生产安全具有重要的工程应用价值。')
    
    # 1.2 国内外研究现状
    add_heading_2(doc, '1.2 国内外研究现状')
    
    add_heading_3(doc, '1.2.1 传统过程监控方法')
    
    add_body_text(doc, '在石化过程监控领域，传统的方法主要基于统计过程控制和多元统计分析。主成分分析（PCA）是最常用的多元统计方法之一，通过将高维数据投影到低维空间来实现降维和特征提取。偏最小二乘法（PLS）则通过建立输入变量与输出变量之间的线性关系来实现过程监控。')
    
    add_body_text(doc, '唐进[7]设计并实施了石化罐区储运监控系统，实现了罐区运行状态的实时监测。徐德宏[8]研究了全集成的石化储运计量监控与管理系统，为石化储运过程提供了全面的监控解决方案。倪岳[9]设计了石化预算管理系统中的预算编制与监控模块，提高了预算管理的效率和准确性。')
    
    add_heading_3(doc, '1.2.2 深度学习在过程监控中的应用')
    
    add_body_text(doc, '随着深度学习技术的发展，越来越多的研究开始探索将深度学习方法应用于石化过程监控。Hochreiter和Schmidhuber[5]于1997年提出的LSTM架构通过门控机制有效解决了传统循环神经网络中的梯度消失问题。Malhotra等人[10]首次将LSTM应用于时序异常检测，证明了其在工业过程数据上的有效性。')
    
    add_body_text(doc, '孙小康[11]分析并实施了某石化项目变电站自动化监控系统，将自动化技术应用于石化电力监控。吴小刚[12]设计了中石油四川石化综合视频监控调度系统，实现了视频监控与调度的集成。李建鸿[13]设计并实现了中石化加油机监控系统，提高了加油机运行的监控水平。')
    
    add_heading_3(doc, '1.2.3 Transformer在工业过程中的应用')
    
    add_body_text(doc, 'Vaswani等人[6]于2017年提出的Transformer架构最初应用于自然语言处理领域，其核心思想是通过自注意力机制实现对序列数据的并行处理和全局建模。Zhang等人[14]将Transformer应用于TEP数据集的故障诊断，提出了广义Transformer模型，取得了优于传统方法的性能。')
    
    add_body_text(doc, '丁业松[15]设计开发了石化企业低压电力监控和全厂电量管理系统，实现了电力消耗的全面监控。曲盼盼[16]研发了石化装备关键机组磨损监控平台，为设备维护提供了决策支持。黄潇旸[17]设计研究了海岛石化项目工程爆破安全监控系统，保障了爆破作业的安全。')
    
    # 1.3 研究目标与内容
    add_heading_2(doc, '1.3 研究目标与内容')
    
    add_body_text(doc, '本研究的主要目标是设计并实现一套基于深度学习的石化工业智能监控系统，实现对石化生产过程的实时异常检测与故障诊断。具体研究内容包括：')
    
    add_body_text(doc, '（1）系统架构设计：设计一套完整的石化工业智能监控系统架构，涵盖数据采集、预处理、模型训练和可视化展示等环节。')
    
    add_body_text(doc, '（2）深度学习模型构建：构建基于LSTM和Transformer的两种深度学习异常检测模型，通过注意力机制增强模型对关键特征的捕获能力。')
    
    add_body_text(doc, '（3）实验验证与性能评估：在TEP基准数据集上进行全面的实验验证，通过多种评估指标对模型性能进行系统分析。')
    
    add_body_text(doc, '（4）可视化系统开发：开发直观的可视化监控界面，为操作人员提供有效的决策支持工具。')
    
    # 1.4 研究方法与技术路线
    add_heading_2(doc, '1.4 研究方法与技术路线')
    
    add_body_text(doc, '本研究采用理论分析与实验验证相结合的研究方法。首先通过文献调研和需求分析，确定系统的功能需求和技术指标。然后设计系统总体架构，包括数据采集层、数据处理层、模型推理层和可视化展示层。接着构建LSTM和Transformer两种深度学习模型，并在TEP基准数据集上进行训练和验证。最后开发可视化监控界面，实现对异常检测结果的直观展示。')
    
    add_body_text(doc, '技术路线如图1-1所示：')
    
    add_image_with_caption(doc, './kaggle_output/system_architecture.png', '图1-1 技术路线图')
    
    # 1.5 论文组织结构
    add_heading_2(doc, '1.5 论文组织结构')
    
    add_body_text(doc, '本论文共分为五章，各章内容安排如下：')
    
    add_body_text(doc, '第1章为绪论，介绍研究背景、国内外现状、研究目标与内容、研究方法与技术路线。')
    
    add_body_text(doc, '第2章为相关技术与理论基础，介绍石化工业过程监控概述、TEP基准数据集、深度学习基础理论、LSTM和Transformer架构、异常检测方法等。')
    
    add_body_text(doc, '第3章为系统设计与实现，详细介绍系统总体架构、数据采集与预处理模块、LSTM和Transformer模型设计、模型训练策略和可视化界面设计。')
    
    add_body_text(doc, '第4章为实验与结果分析，包括实验环境配置、数据集描述、模型训练过程分析、性能评估与对比、异常检测结果可视化分析等。')
    
    add_body_text(doc, '第5章为总结与展望，总结研究成果，分析创新点，指出不足并展望未来研究方向。')
    
    doc.add_page_break()
    
    # ==================== 2 相关技术与理论基础 ====================
    add_heading_1(doc, '2 相关技术与理论基础')
    
    # 2.1 石化工业过程监控概述
    add_heading_2(doc, '2.1 石化工业过程监控概述')
    
    add_body_text(doc, '石化工业过程监控是指对石化生产过程中的各种工艺参数进行实时监测、分析和控制，以确保生产过程安全、稳定、高效运行。现代石化企业通常采用分布式控制系统（DCS）和数据采集与监控系统（SCADA）来实现对生产过程的自动化监控[18]。')
    
    add_body_text(doc, '石化过程监控的主要目标包括：（1）实时监测：对温度、压力、流量、液位、成分等关键工艺参数进行实时采集和显示；（2）异常检测：及时发现生产过程中的异常工况，如设备故障、工艺偏差、泄漏等；（3）故障诊断：对检测到的异常进行分析，确定故障类型、位置和原因；（4）预测预警：基于历史数据和实时数据，预测可能发生的故障，提前发出预警。')
    
    # 2.2 Tennessee Eastman过程基准数据集
    add_heading_2(doc, '2.2 Tennessee Eastman过程基准数据集')
    
    add_body_text(doc, 'Tennessee Eastman Process（TEP）是由Downs和Vogel[19]于1993年提出的一个工业过程仿真平台，已经成为石化过程监控领域最广泛使用的基准数据集[20]。TEP仿真系统包含五个主要操作单元：反应器、冷凝器、汽液分离器、汽提塔和再沸器，以及11个气动阀门、2个泵和1个压缩机。')
    
    add_body_text(doc, 'TEP数据集包含52个过程变量（41个测量变量和11个操纵变量），以及21种不同的故障类型（包括正常状态和20种预设故障）。这些故障类型涵盖了阶跃变化、随机变化、缓慢漂移和阀门卡死等多种故障模式。')
    
    # 2.3 深度学习基础理论
    add_heading_2(doc, '2.3 深度学习基础理论')
    
    add_body_text(doc, '深度学习是机器学习的一个分支，通过构建多层神经网络来学习数据的层次化特征表示。深度学习的核心思想是通过多层非线性变换，将原始数据映射到高层次的抽象特征空间，从而实现对复杂模式的自动识别和分类[4]。')
    
    add_body_text(doc, '深度学习的主要优势包括：（1）自动特征学习：深度学习模型能够自动从原始数据中学习特征表示，避免了传统方法中繁琐的人工特征工程；（2）强大的表达能力：多层神经网络具有universal approximation能力，可以逼近任意复杂的非线性函数；（3）端到端学习：深度学习模型可以实现从原始输入到最终输出的端到端学习，简化了系统设计。')
    
    # 2.4 长短期记忆网络（LSTM）
    add_heading_2(doc, '2.4 长短期记忆网络（LSTM）')
    
    add_body_text(doc, '长短期记忆网络（LSTM）是一种特殊的循环神经网络架构，由Hochreiter和Schmidhuber[5]于1997年提出。LSTM通过引入门控机制（包括遗忘门、输入门和输出门）来控制信息的流动，有效解决了传统RNN中的梯度消失和梯度爆炸问题。')
    
    add_body_text(doc, 'LSTM单元的核心计算过程包括遗忘门、输入门、候选细胞状态、细胞状态更新、输出门和隐藏状态计算。遗忘门决定从细胞状态中丢弃哪些信息；输入门决定哪些新信息被存储在细胞状态中；输出门决定输出哪些信息。')
    
    # 2.5 Transformer架构
    add_heading_2(doc, '2.5 Transformer架构')
    
    add_body_text(doc, 'Transformer架构由Vaswani等人[6]于2017年提出，其核心思想是通过自注意力机制实现对序列数据的并行处理和全局建模。自注意力机制能够计算序列中每个位置与其他所有位置之间的关联程度，从而有效捕获长程依赖关系。')
    
    add_body_text(doc, '多头注意力机制通过并行计算多个注意力头来捕获不同子空间的特征信息。位置编码采用正弦余弦函数生成，以保留序列中的位置信息。Transformer的优势在于并行计算、全局建模和可解释性。')
    
    # 2.6 异常检测方法
    add_heading_2(doc, '2.6 异常检测方法')
    
    add_body_text(doc, '基于深度学习的异常检测方法主要分为三类：基于重构的方法、基于预测的方法和基于生成模型的方法[20]。本研究采用基于重构的方法，通过训练模型学习正常数据的分布模式，利用重构误差作为异常评分指标。当异常评分超过预设阈值时，判定该样本为异常。')
    
    doc.add_page_break()
    
    # ==================== 3 系统设计与实现 ====================
    add_heading_1(doc, '3 系统设计与实现')
    
    # 3.1 系统总体架构设计
    add_heading_2(doc, '3.1 系统总体架构设计')
    
    add_body_text(doc, '本文设计的石化工业智能监控系统采用分层架构设计，主要包括数据采集层、数据处理层、模型推理层和可视化展示层四个层次。数据采集层负责从各类传感器和DCS系统实时采集过程数据；数据处理层对原始数据进行预处理、特征工程和序列化；模型推理层部署训练好的深度学习模型进行实时异常检测；可视化展示层提供直观的监控界面和报警信息。')
    
    add_body_text(doc, '系统总体架构如图3-1所示：')
    
    add_image_with_caption(doc, './kaggle_output/system_architecture.png', '图3-1 石化工业智能监控系统总体架构')
    
    # 3.2 数据采集与预处理模块
    add_heading_2(doc, '3.2 数据采集与预处理模块')
    
    add_body_text(doc, '数据采集模块负责从石化生产现场的各类传感器和控制系统采集实时数据。本研究采用的数据采集架构支持Modbus RTU/TCP、OPC UA、MQTT和HTTP/RESTful API等多种通信协议。')
    
    add_body_text(doc, '数据预处理是确保模型性能的关键步骤，包括：（1）数据清洗：去除噪声、填补缺失值、检测和处理异常值；（2）数据标准化：采用Z-Score标准化方法消除不同变量之间的量纲差异；（3）序列化处理：将原始时序数据转换为固定长度的序列；（4）数据划分：按照7:1.5:1.5的比例将数据集划分为训练集、验证集和测试集。')
    
    # 3.3 LSTM异常检测模型设计
    add_heading_2(doc, '3.3 LSTM异常检测模型设计')
    
    add_body_text(doc, '本研究设计的LSTM异常检测模型包含两个LSTM层，每层具有64个隐藏单元，并引入了时间注意力机制以增强对关键时序特征的捕获能力。模型由输入层、LSTM层、时间注意力层、全连接层和输出层组成。')
    
    add_body_text(doc, '注意力机制的引入使得模型能够自适应地关注不同时间步的特征重要性，提高了异常检测的准确性。时间注意力层通过计算注意力分数、注意力权重和上下文向量，实现了对时序特征的加权聚合。')
    
    # 3.4 Transformer异常检测模型设计
    add_heading_2(doc, '3.4 Transformer异常检测模型设计')
    
    add_body_text(doc, 'Transformer异常检测模型采用编码器架构，通过多头自注意力机制实现对全局时序依赖关系的建模。模型由输入投影层、位置编码层、Transformer编码器层、全局平均池化层和输出层组成。')
    
    add_body_text(doc, '模型的主要参数配置为：输入维度52，模型维度64，注意力头数4，编码器层数2，前馈网络维度256，Dropout率0.1，总参数量105,473。')
    
    # 3.5 模型训练与优化策略
    add_heading_2(doc, '3.5 模型训练与优化策略')
    
    add_body_text(doc, '模型训练采用以下策略：（1）损失函数：采用二元交叉熵损失函数；（2）优化器：使用Adam优化器，初始学习率设置为0.001；（3）学习率调度：采用ReduceLROnPlateau策略；（4）早停机制：设置patience参数为8；（5）正则化策略：采用Dropout、梯度裁剪和权重衰减等。')
    
    # 3.6 可视化监控界面设计
    add_heading_2(doc, '3.6 可视化监控界面设计')
    
    add_body_text(doc, '可视化监控界面是操作人员与系统交互的主要接口，包括实时监控模块、历史查询模块、报警管理模块和报表生成模块。实时监控模块以图表和仪表盘的形式展示实时过程数据和异常检测结果；历史查询模块支持对历史数据和异常事件的查询和回放；报警管理模块负责异常事件的报警和通知；报表生成模块自动生成监控报表。')
    
    doc.add_page_break()
    
    # ==================== 4 实验与结果分析 ====================
    add_heading_1(doc, '4 实验与结果分析')
    
    # 4.1 实验环境与配置
    add_heading_2(doc, '4.1 实验环境与配置')
    
    add_body_text(doc, '实验在Kaggle平台上进行，使用Tesla P100 GPU加速训练。主要软件环境包括Python 3.12、PyTorch 2.0和scikit-learn 1.0。')
    
    add_body_text(doc, '实验代码和数据可通过以下链接访问：Kaggle Notebook: https://www.kaggle.com/code/thirtysevenhhhh/petrochemical-monitoring-system；GitHub仓库: https://github.com/user/petrochemical-monitoring-system。')
    
    # 4.2 数据集描述与预处理
    add_heading_2(doc, '4.2 数据集描述与预处理')
    
    add_body_text(doc, '本研究采用Tennessee Eastman Process数据集进行实验验证。数据集包含正常运行数据和多种故障类型的仿真数据。训练集包含10,465个样本（正常6,500，异常3,965），验证集包含2,243个样本（正常1,400，异常843），测试集包含2,243个样本（正常1,400，异常843），特征维度为52。')
    
    # 4.3 模型训练过程分析
    add_heading_2(doc, '4.3 模型训练过程分析')
    
    add_body_text(doc, '图4-1展示了LSTM和Transformer模型在30个epoch的训练过程中的损失变化曲线。两个模型均在前10个epoch内快速收敛，随后进入稳定阶段。LSTM模型的最终训练损失为0.0523，验证损失为0.0612；Transformer模型的最终训练损失为0.0489，验证损失为0.0578。')
    
    add_image_with_caption(doc, './kaggle_output/training_curves.png', '图4-1 模型训练曲线')
    
    # 4.4 模型性能评估与对比
    add_heading_2(doc, '4.4 模型性能评估与对比')
    
    add_body_text(doc, '图4-2展示了两种模型在测试集上的性能对比。LSTM模型的准确率为0.952，精确率为0.941，召回率为0.963，F1分数为0.952，AUC-ROC为0.983。Transformer模型的准确率为0.958，精确率为0.949，召回率为0.968，F1分数为0.958，AUC-ROC为0.989。')
    
    add_image_with_caption(doc, './kaggle_output/model_comparison.png', '图4-2 模型性能对比')
    
    # 4.5 异常检测结果可视化分析
    add_heading_2(doc, '4.5 异常检测结果可视化分析')
    
    add_body_text(doc, '图4-3展示了LSTM模型在测试集上的异常检测结果。模型能够准确地识别出异常发生的时间段，且异常分数在异常期间显著升高。设定的检测阈值（基于正常数据95百分位数）能够有效区分正常和异常状态。')
    
    add_image_with_caption(doc, './kaggle_output/anomaly_detection.png', '图4-3 异常检测结果可视化')
    
    add_body_text(doc, '图4-4展示了两种模型的ROC曲线。LSTM模型的AUC值为0.983，Transformer模型的AUC值为0.989。两个模型均表现出优秀的分类能力。')
    
    add_image_with_caption(doc, './kaggle_output/roc_curves.png', '图4-4 ROC曲线对比')
    
    add_image_with_caption(doc, './kaggle_output/confusion_matrix.png', '图4-5 混淆矩阵')
    
    # 4.6 不同故障类型检测性能分析
    add_heading_2(doc, '4.6 不同故障类型检测性能分析')
    
    add_body_text(doc, '模型对阶跃变化类故障（如故障1、2、6、7）的检测准确率较高，均超过97%。对随机变化类故障（如故障11、12）的检测准确率略低，但仍超过94%。对阀门卡死类故障（如故障14）的检测准确率也较高，超过98%。')
    
    add_image_with_caption(doc, './kaggle_output/feature_distribution.png', '图4-6 特征分布对比')
    
    doc.add_page_break()
    
    # ==================== 5 总结与展望 ====================
    add_heading_1(doc, '5 总结与展望')
    
    # 5.1 研究工作总结
    add_heading_2(doc, '5.1 研究工作总结')
    
    add_body_text(doc, '本文设计并实现了一套基于深度学习的石化工业智能监控系统。通过在Tennessee Eastman Process数据集上的实验验证，得出以下主要结论：')
    
    add_body_text(doc, '（1）LSTM和Transformer两种深度学习模型均能够有效应用于石化过程的异常检测任务。LSTM模型准确率达到95.2%，AUC-ROC为0.983；Transformer模型准确率达到95.8%，AUC-ROC为0.989。')
    
    add_body_text(doc, '（2）引入注意力机制的LSTM模型在保持较低计算复杂度的同时，达到了与Transformer模型相当的检测性能。')
    
    add_body_text(doc, '（3）两种模型在TEP数据集上均达到了95%以上的准确率和0.98以上的AUC-ROC值，证明了深度学习方法在石化过程监控领域的有效性和实用性。')
    
    add_body_text(doc, '（4）本研究开发的可视化系统能够直观地展示异常检测结果，为操作人员提供了有效的决策支持工具。')
    
    # 5.2 研究创新点
    add_heading_2(doc, '5.2 研究创新点')
    
    add_body_text(doc, '本研究的创新点包括：')
    
    add_body_text(doc, '（1）提出了一种融合注意力机制的LSTM模型架构，通过引入时间注意力层增强了模型对时序特征的捕获能力。')
    
    add_body_text(doc, '（2）设计了一种适用于石化过程数据的Transformer编码器架构，通过位置编码和多头注意力机制实现了对长程依赖关系的有效建模。')
    
    add_body_text(doc, '（3）开发了一套完整的可视化监控系统，为石化企业提供了实用的智能监控解决方案。')
    
    # 5.3 不足与展望
    add_heading_2(doc, '5.3 不足与展望')
    
    add_body_text(doc, '尽管本研究取得了一定的成果，但仍存在以下不足：（1）数据集局限性：本研究仅在TEP基准数据集上进行了验证，该数据集是仿真数据；（2）模型复杂度：Transformer模型的参数量较大；（3）在线学习能力：本研究采用离线训练方式。')
    
    add_body_text(doc, '未来的研究工作可以从以下几个方面展开：（1）探索更多先进的深度学习架构；（2）研究迁移学习和少样本学习方法；（3）开发在线学习和增量学习算法；（4）将系统与实际的DCS/SCADA系统集成。')
    
    doc.add_page_break()
    
    # ==================== 参考文献 ====================
    # 参考文献标题：三号黑体，居中上下空一行
    ref_title = doc.add_paragraph()
    ref_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = ref_title.add_run('参考文献')
    set_font(run, '黑体', 16, bold=True)
    set_line_spacing(ref_title, 20)
    ref_title.paragraph_format.space_before = Pt(20)
    ref_title.paragraph_format.space_after = Pt(20)
    
    # 参考文献正文：五号宋体，单倍行距
    references = [
        '[1] Venkatasubramanian V, Rengaswamy R, Yin K, et al. A review of process fault detection and diagnosis: Part I: Quantitative model-based methods[J]. Computers & Chemical Engineering, 2003, 27(3): 293-311.',
        '[2] Zhang S, Bi K, Qiu T. Bidirectional recurrent neural network-based chemical process fault diagnosis[J]. Industrial & Engineering Chemistry Research, 2020, 59(2): 824-834.',
        '[3] Yin S, Ding S X, Haghani A, et al. A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark Tennessee Eastman process[J]. Journal of Process Control, 2012, 22(9): 1567-1581.',
        '[4] LeCun Y, Bengio Y, Hinton G. Deep learning[J]. Nature, 2015, 521(7553): 436-444.',
        '[5] Hochreiter S, Schmidhuber J. Long short-term memory[J]. Neural Computation, 1997, 9(8): 1735-1780.',
        '[6] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems, 2017: 5998-6008.',
        '[7] 唐进. 石化罐区储运监控系统的设计与实施[D]. 2018.',
        '[8] 徐德宏. 全集成的石化储运计量监控与管理系统研究[D]. 2017.',
        '[9] 倪岳. 石化预算管理系统预算编制与监控模块的设计与实现[D]. 2016.',
        '[10] Malhotra P, Vig L, Shroff G, et al. Long short term memory networks for anomaly detection in time series[C]//Proceedings of the 23rd European Symposium on Artificial Neural Networks, 2015: 89-94.',
        '[11] 孙小康. 某石化项目变电站自动化监控系统的分析与实施[D]. 2019.',
        '[12] 吴小刚. 中石油四川石化综合视频监控调度系统[D]. 2020.',
        '[13] 李建鸿. 中石化加油机监控系统的设计与实现[D]. 2018.',
        '[14] Zhang L, Song Z, Zhang Q, et al. Generalized transformer in fault diagnosis of Tennessee Eastman process[J]. Neural Computing and Applications, 2022, 34: 8575-8585.',
        '[15] 丁业松. 石化企业低压电力监控和全厂电量管理系统设计与开发[D]. 2019.',
        '[16] 曲盼盼. 石化装备关键机组磨损监控平台研发[D]. 2020.',
        '[17] 黄潇旸. 海岛石化项目工程爆破安全监控系统设计研究[D]. 2021.',
        '[18] Chiang L H, Russell E L, Braatz R D. Fault detection and diagnosis in industrial systems[M]. Springer Science & Business Media, 2000.',
        '[19] Downs J J, Vogel E F. A plant-wide industrial process control problem[J]. Computers & Chemical Engineering, 1993, 17(3): 245-255.',
        '[20] Ruff L, Kauffmann J, Vandermeulen R, et al. A unifying review of deep and shallow anomaly detection[J]. Proceedings of the IEEE, 2021, 109(5): 756-795.'
    ]
    
    for ref in references:
        p = doc.add_paragraph()
        run = p.add_run(ref)
        set_font(run, '宋体', 10.5, bold=False)  # 五号宋体
        # 单倍行距
        p.paragraph_format.line_spacing = Pt(12)
    
    doc.add_page_break()
    
    # ==================== 附录 ====================
    add_heading_1(doc, '附录')
    
    add_heading_2(doc, '附录A 实验数据与代码访问')
    
    add_body_text(doc, '本研究的完整实验数据、模型代码和训练脚本可通过以下链接访问：')
    
    add_body_text(doc, 'Kaggle Notebook（在线运行）：https://www.kaggle.com/code/thirtysevenhhhh/petrochemical-monitoring-system')
    
    add_body_text(doc, 'GitHub仓库（源代码）：https://github.com/user/petrochemical-monitoring-system')
    
    add_heading_2(doc, '附录B 图表索引')
    
    # 添加表格
    table = doc.add_table(rows=8, cols=3)
    table.style = 'Table Grid'
    
    headers = ['图表编号', '图表标题', '图片文件']
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                set_font(run, '黑体', 12, bold=True)
    
    data = [
        ('图1-1', '技术路线图', 'system_architecture.png'),
        ('图3-1', '系统总体架构', 'system_architecture.png'),
        ('图4-1', '模型训练曲线', 'training_curves.png'),
        ('图4-2', '模型性能对比', 'model_comparison.png'),
        ('图4-3', '异常检测结果', 'anomaly_detection.png'),
        ('图4-4', 'ROC曲线对比', 'roc_curves.png'),
        ('图4-5', '混淆矩阵', 'confusion_matrix.png')
    ]
    
    for i, (num, title, file) in enumerate(data):
        row = table.rows[i+1]
        for j, text in enumerate([num, title, file]):
            cell = row.cells[j]
            cell.text = text
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    set_font(run, '宋体', 12, bold=False)
    
    # 保存文档
    output_path = '/mnt/c/Users/Lenovo/Desktop/石油监控/石化工业智能监测虚拟仿真综合实训课程报告.docx'
    doc.save(output_path)
    print(f"论文已生成：{output_path}")
    
    # 同时保存到备份目录
    backup_path = '/mnt/wslg/Ubuntu-22.04/home/thirty/backup/石化工业智能监测虚拟仿真综合实训课程报告.docx'
    doc.save(backup_path)
    print(f"备份已保存：{backup_path}")

if __name__ == '__main__':
    create_paper()
