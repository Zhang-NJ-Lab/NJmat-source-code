# from chemdataextractor.doc import Document
# #from chemdataextractor.doc.document
# from chemdataextractor.model.units import TemperatureModel, RatioModel,TimeModel,DimensionlessModel
# from chemdataextractor.model.model import Compound, ModelType, StringType, CurieTemperature
# from chemdataextractor.model.base import BaseModel
# #from chemdataextractor.model.pv_model import OpenCircuitVoltage
# from chemdataextractor.parse.elements import I,R,W,Optional,ZeroOrMore
# from chemdataextractor.parse.actions import join, merge
# from chemdataextractor.parse.cem import CompoundParser,ChemicalLabelParser
# from chemdataextractor.reader.plaintext import PlainTextReader
#

import pandas as pd
from chemdataextractor import Document

# 读取CSV文件
df = pd.read_csv('C:\\Users\\DELL\\Desktop\\photovoltaic_cosine_similarity.csv')

# 提取第一列数据
column_data = df[0]

# 定义函数来识别化学物质
def is_chemical(entity):
    doc = Document(entity)
    chem_entities = doc.cems
    return len(chem_entities) > 0

# 创建新列来表示化学物质
df['Chemical'] = column_data.apply(lambda x: 1 if is_chemical(x) else 0)

# 保存结果到新的CSV文件
df.to_csv('C:\\Users\\DELL\\Desktop\\output_file.csv', index=False)




# with open('1.txt', 'rb') as f:
#     doc = Document.from_file(f)
# #a = open('3.txt', 'a', encoding='UTF8')
#
# print(doc.cems)


# doc.records
# '''
# class Additives(BaseModel):
#     specifier_expr = I('Additives') .add_action(join)
#
#     name=StringType()
#     parsers = [ChemicalLabelParser()]
# '''
#
# class PowerConversionEfficiency(RatioModel):
#     #specifier_expr = ((I('Power')+I('Conversion')+I('Efficiency')) | I('PCE')).add_action(join)
#
#     #specifier_expr = (((I('Power')|I('photo')|I('photovoltaic'))+ I('Conversion') + I('Efficiency')+I('(PCE)')+(I('is')|I('of')|I('was')))|(I('improved')+I('PCE')+I('of'))|((I('the')|I('this'))+I('PCE')+(I('is')|I('of')))).add_action(join)
#     #specifier_expr = (((I('Power')|I('photo')|I('photovoltaic'))+I('Conversion')+I('Efficiency')+(W('(')+I('PCE')+W(')')).add_action(merge)+(I('is')|I('of')|I('was')))|((I('the')|I('this'))+I('PCE')+(I('is')|I('of')))).add_action(join)
#     specifier_expr =(I('Power')+I('Conversion') + I('Efficiency')|I('PCE')|I('photo')+I('conversion')+I('efficiency')).add_action(join)
#     specifier = StringType(parse_expression=specifier_expr, required=True, contextual=True, updatable=True)
#     compound = ModelType(Compound, required=True, contextual=True)
#
# class PowerConversionEfficiency1(RatioModel):
#     specifier_expr = (ZeroOrMore(I('A')|I('the')).hide()+ZeroOrMore(I('high')|I('highest'))+ZeroOrMore(I('average'))+I('Power') + I('Conversion') + I('Efficiency') +(W('(')+I('PCE')+W(')')).add_action(merge)+(W('=')|I('of')|I('is')|I('was')|I('at'))).add_action(join)
#     specifier = StringType(parse_expression=specifier_expr, required=True, contextual=True, updatable=True)
#     compound = ModelType(Compound, required=True, contextual=True)
#     #dditives=ModelType(Additives, required=False, contextual=False)
# class PowerConversionEfficiency2(RatioModel):
#     specifier_expr = (ZeroOrMore(I('A')|I('the')).hide()+(I('Power') + I('Conversion') + I('Efficiency') |I('PCE'))+(I('of')|I('is')|I('was')|I('at')|I('are')|I('were'))+I('improved')+I('from')+(R(r'\d+')+W('%')).add_action(merge)+I('to')).add_action(join)
#     specifier = StringType(parse_expression=specifier_expr, required=True, contextual=True, updatable=True)
#     compound = ModelType(Compound, required=True, contextual=True)
# #class FromTo(RatioModel):
# f = open('1.txt', 'rb')
# doc = Document.from_file(f)
# #doc.models = [PowerConversionEfficiency,PowerConversionEfficiency1]
# doc.models = [PowerConversionEfficiency,PowerConversionEfficiency1,PowerConversionEfficiency2]
# print(doc.models)
# for record in doc.records:
#     print(record.serialize())
#    #print(record.serialize(),file=a)
#
