"""Deterministic fabricated everyday scenes. No real chat sources."""
from datetime import date, timedelta
import json
from pathlib import Path

root = Path(__file__).resolve().parent
topics = [('栀子花','占位甲在阳台种了栀子花，每周三检查盆土。'),('南瓜粥','占位乙周末打算煮南瓜粥，购物清单还缺小米。'),('蓝色雨伞','占位甲把蓝色雨伞放在图书馆入口的架子上。'),('自行车','占位乙预约周五去修自行车的车铃。'),('天文馆','占位丙计划下周六去天文馆看星空展览。'),('散步','占位丁晚饭后沿着虚构河岸散步。'),('拼图','占位甲今天整理了一盒森林拼图。')]
messages = []
for day in range(21):
    for slot in range(15):
        topic,body = topics[(day*15+slot)%len(topics)]
        messages.append(dict(ns='fixture',id=f'm{day*15+slot+1:04}',time=f'{date(2026,10,1)+timedelta(days=day)}T{8+slot//2:02}:{30*(slot%2):02}:00+00:00',speaker=f'占位{"甲乙丙丁"[slot%4]}',text=f'合成第{day+1}天第{slot+1}则：{body}'))
(root/'messages.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in messages))
notes = [dict(id=f'n{i+1:02}',date=f'2026-10-{i*3+1:02}',written_at=f'2026-10-{i*3+2:02}T01:00:00+00:00',title=f'合成日印象{i+1}',body='\n\n'.join(topics[(i+j)%7][1] for j in range(4)),cues=[topics[i][0]]) for i in range(6)]
(root/'impressions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in notes))
queries = [dict(query=q,answer_ids=[f'm{i+1:04}']) for i,q in enumerate(['栀子花什么时候检查盆土？','煮南瓜粥还缺什么？','蓝色雨伞放在哪里？','自行车哪天修车铃？','天文馆看什么展览？'])]
(root/'queries.json').write_text(json.dumps(queries,ensure_ascii=False,indent=2)+'\n')
