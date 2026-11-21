"""Produce analysis directly from the measured notebook state."""
from __future__ import annotations
from pathlib import Path
import json

PDP = 'https://old.adilet.zan.kz/rus/docs/Z1300000094/compare/eng'
OECD = 'https://oecd.ai/en/ai-principles'
AI_LAW = 'https://old.adilet.zan.kz/eng/docs/Z2500000230'

def describe_baseline(table):
    parts=[]
    for attr, floor in [('name_language','null_name'), ('gender_coded_name','null_name'), ('university','null_university')]:
        s,f=table[attr],table[floor]
        lo,hi=s['flip_rate_ci95']; flo,fhi=f['flip_rate_ci95']
        overlap=max(lo,flo)<=min(hi,fhi)
        parts.append(f"{attr}: {s['flips']}/{s['n']} flips ({s['flip_rate']:.1%}; 95% CI {lo:.1%}–{hi:.1%}), "
                     f"floor {floor}: {f['flips']}/{f['n']} ({f['flip_rate']:.1%}; {flo:.1%}–{fhi:.1%}); "
                     f"mean |delta| {s['mean_abs_score_diff']:.2f} vs floor {f['mean_abs_score_diff']:.2f}. "
                     + ('Интервалы flip rates перекрываются: уверенного превышения floor здесь не установлено.' if overlap else
                        'Интервалы flip rates не перекрываются; это описательный сигнал, не формальный тест разности rates.'))
    return '\n'.join(parts)

def make_register(cls, observation):
    law='РК, Закон № 94-V от 21.05.2013 «О персональных данных и их защите»; '+PDP
    ai_law='РК, Закон № 230-VIII от 17.11.2025 «Об искусственном интеллекте»; '+AI_LAW
    source='OECD Recommendation on AI, 2019, updated 2024 (non-binding); '+OECD
    rows=[
        ('validate_input','Robustness, security and safety','1.4',ai_law,'ст. 15(2)(1)–(2), 18(1); конкретные диапазоны — политика прототипа',
         'Baseline использует чистые профили; этот риск проверяется отдельно невозможными значениями в Part 4.'),
        ('blind','Human rights and democratic values, including fairness and privacy','1.2',
         ai_law,'ст. 6; blinding — выбранная реализация принципа равенства',
         'Ограничиваю доступ к нерелевантным полям; не приписываю нулевые after flips исправлению модели. '+observation),
        ('refuse','Human rights and democratic values, including fairness and privacy','1.2',law,
         'ст. 7(1), 19-1(1); дополнительные возрастные/текстовые ограничения — политика прототипа',
         'Чистая baseline-выборка не проверяет недопустимые запросы; Part 4 отдельно проверяет пять запретов до inference.'),
        ('log','Accountability','1.5',ai_law,'ст. 15(2)(3), 16(1)(4)–(5); JSONL — выбранный формат',
         'Сохраняю различие model decision и final outcome, чтобы объяснять изменения отдельного профиля и причины отказов.'),
        ('escalate','Transparency and explainability; human agency and oversight','1.3; 1.2',law,
         'ст. 19-1(2)–(3); автоматическая эскалация — выбранная мера, не полная реализация процедуры возражений',
         'Band выбирается из baseline-распределения; фактическая нагрузка после blinding измеряется заново. Ошибка парсинга не должна становиться скрытым отказом.')]
    return [cls(control=a,principle=b,principle_source=source+' §'+c,instrument=d,provision=e,
                why_here=f+' Синтетический эксперимент: ориентир для реального фонда в РК, не заявление о полной правовой совместимости.')
            for a,b,c,d,e,f in rows]

def table_md(table):
    lines=['| Атрибут | Flips / n | Rate [95% Wilson CI] | a→b / b→a | McNemar p | Разные scores / scored | Mean b−a [95% CI] | Mean abs [95% CI] |',
           '|---|---:|---|---:|---:|---:|---|---|']
    for name,s in table.items():
        lo,hi=s['flip_rate_ci95']; sl,sh=s['ci95_score_diff']; al,ah=s['ci95_abs_score_diff']
        lines.append(f"| {name} | {s['flips']}/{s['n']} | {s['flip_rate']:.1%} [{lo:.1%}, {hi:.1%}] | "
                     f"{s['a_to_b']}/{s['b_to_a']} | {s['mcnemar_exact_p']:.3f} | {s['n_score_diff']}/{s['n_scored']} | "
                     f"{s['mean_score_diff']:+.2f} [{sl:+.2f}, {sh:+.2f}] | {s['mean_abs_score_diff']:.2f} [{al:.2f}, {ah:.2f}] |")
    return '\n'.join(lines)

def preserve_personal_section(previous, generated):
    """Preserve the author's AI-use text verbatim, including whitespace."""
    heading = '## AI-use declaration'
    if heading not in previous:
        raise ValueError('Keep the personal AI-use section in JUSTIFICATION.md')
    return generated.rstrip() + '\n\n' + heading + previous.split(heading, 1)[1]


def write_reports(g):
    from collections import Counter
    from dataclasses import asdict
    import platform
    import numpy
    real = g.get("RUN_REAL_MODEL", True)
    if not real or g['llm'].model_id != 'Qwen/Qwen2.5-3B-Instruct':
        raise ValueError('Submission reports require a real Qwen run')
    import torch, transformers
    versions = f"torch {torch.__version__}, transformers {transformers.__version__}"
    b,a=g['BASE'],g['AFTER']; bt,at=g['BASE_TABLE'],g['AFTER_TABLE']
    n=len(a.verdicts); changes=g['profile_changes'](b,a)
    band=g['BORDERLINE']; planned,scored=g['band_share'](g['BASE_SCORES'],*band)
    escalated=sum(v.outcome=='escalate' for v in a.verdicts)
    reasons=dict(Counter(v.reason_code for v in a.verdicts))
    oriented_rows=[]; oriented_data={}
    for attr,first,key,label in g['ORIENTATIONS']:
        s=g['oriented'](g['PAIRS'],b.outcomes_a,b.outcomes_b,attr,first,key)
        oriented_data[label]=s
        oriented_rows.append(f"| {label} | {s['n']} | {s['a_to_b']} | {s['b_to_a']} | {s['mcnemar_exact_p']:.3f} | {g['gap'](s)} | {s['sign_test_p']:.3f} |")
    orientation='\n'.join(['| Первая группа / другая | n | Первая shortlisted, другая нет | Наоборот | McNemar p | Score первой − другой [95% CI] | Sign p |', '|---|---:|---:|---:|---:|---|---:|']+oriented_rows)
    requests='\n'.join(['| Запрос | Bare | Harness | Reason | Вызовов harness |','|---|---|---|---|---:|']+
        [f"| {r['request']} | {r['bare']} | {r['harness']} | {r['reason_code']} | {r['model_calls']} |" for r in g['REQUEST_RESULTS']])
    log=g['read_log'](g['LOG_PATH'])
    examples=[]
    for outcome in ('shortlist','reject','escalate','refuse'):
        rec=next((r for r in log if r['outcome']==outcome),None)
        if rec: examples.append(f"- `{rec['record_id']}`: pid `{rec['pid']}`, model decision `{rec['decision']}`, score `{rec['score']}`, outcome `{rec['outcome']}`, reason `{rec['reason_code']}`.")
    costs='\n'.join(['| Конфигурация | Профилей | Вызовов | Prompt tokens | Output tokens | Время model calls, s |','|---|---:|---:|---:|---:|---:|']+
        [f'| {r.label} | {len(r.verdicts)} | {r.calls} | {r.prompt_tokens} | {r.new_tokens} | {r.seconds:.1f} |' for r in (b,a)])
    report=f'''# REPORT — what the measurement showed, who it affects, and whose responsibility it is

## 1. Measurement read-out

Выполнен прогон Qwen/Qwen2.5-3B-Instruct, greedy decoding, prompt v1,
seed {g['SEED']}, {g['N_PAIRS']} пар на атрибут: всего {len(g['PAIRS'])} пар / {n} профилей.
Данные полностью синтетические. Среда: Python {platform.python_version()}, {versions},
numpy {numpy.__version__}, {g['llm'].device}.
Используется float16; расхождения между GPU не измерялись.
Merit-поля не менялись между before и after.

### Before

{table_md(bt)}

Здесь a→b означает shortlist у a и не-shortlist у b. Роли a/b произвольны, поэтому направление
социального различия читаю только по переориентированной таблице ниже.

{orientation}

{describe_baseline(bt)}

Сравнение с floor описательное: перекрытие интервалов не является тестом равенства двух rates,
а неперекрытие не заменяет заранее выбранную процедуру множественных сравнений.
McNemar проверяет баланс направлений внутри пар, а не разность с floor. Sign test игнорирует
нулевые score differences. Все p-values исследовательские, без поправки за множественность.
Bootstrap CI средних использует 2000 перевыборок; фиксированный seed делает сводку воспроизводимой,
но не устраняет ограниченность выборки. Signed mean может компенсировать разнонаправленные сдвиги,
поэтому mean absolute difference и число изменившихся баллов обязательны.

При {g['N_PAIRS']} парах минимальное значимое число discordant pairs в одном направлении,
если противоположных нет: {g['min_detectable'](g['N_PAIRS'])} (двусторонний exact test, alpha=0.05).
При одном противоположном исходе нужно {g['min_detectable'](g['N_PAIRS'],1)}, при двух — {g['min_detectable'](g['N_PAIRS'],2)}.
Это порог достижимости p-value, не полноценный power analysis. Языковые ориентированные
подвыборки ещё меньше. Нулевой результат не доказывает отсутствие эффекта.

`null_name` меняет имя внутри той же языковой/гендерной группы, `null_university` — вуз внутри
того же tier. Это чувствительность к нерелевантным деталям, не оценка всей неопределённости.
Оба prompt в паре имеют один порядок строк. Порядок, иной system prompt, другая модель и
реальные анкеты не исследованы; optional order probe не выполнялся. Для усиления вывода нужны
независимые seeds, больше пар, несколько вариантов prompt и отдельный order probe.

### After и цена контролей

{table_md(at)}

Нулевые различия после скрытия имени и вуза ожидаются по построению: два входных текста
совпадают. Это проверка ограничения информации, не доказательство исправления модели.
Изменилось итоговое решение у {changes['outcome_changed']}/{n} профилей; обе оценки имеются у
{changes['both_scored']}, балл изменился у {changes['score_changed']}; средний абсолютный сдвиг
**среди изменившихся**: {changes['mean_abs_delta']}. Эти изменения нельзя однозначно приписать
использованию удалённого признака: удаление строк также меняет форму prompt.

Band {band} выбран до after по midpoint(reject Q90, shortlist Q10) ±2.
План по baseline: {planned}/{scored} ({planned/scored:.1%}); фактически отправлено человеку
{escalated}/{n} ({escalated/n:.1%}), ориентир нагрузки — 25%. Reasons: `{reasons}`.
Band не подгонялся по итогам after; расхождение — измеренный предел переноса калибровки.

{costs}

Таблица затрат относится к двум полным измерениям. Демонстрационный ответ и шесть bare-сценариев
Part 4 считаются отдельно. Всего за ноутбук: {g['MODEL'].calls} model calls.

{requests}

Ложных отказов на чистой выборке: {len(g['wrongly'])}/{n}. Это не гарантия для других анкет.
Каждый refusal произошёл до model call. В итоговом журнале {len(log)} записей:
{dict(Counter(r['outcome'] for r in log))}.

## 2. Who is affected

Ориентированная таблица связывает конкретные языковые группы имён, гендерную кодировку и tier
университета с числами discordant pairs. Внутри такой пары заслуги одинаковы; сторона без
shortlist теряет доступ к следующему этапу относительно своего matched counterpart.
Однако в данных нет ground truth о заслуженной стипендии: я не могу назвать любой reject
ошибочным и не выдаю число flips за число дискриминированных реальных людей.
Язык имени — не измерение этничности, кодировка имени — не установленный гендер человека.

Потенциально неверный reject стоит заявителю финансирования, времени на обжалование и
возможности учиться. Потенциально неверный shortlist расходует ограниченные места и время
комиссии, может вытеснить другого нуждающегося. Риски этих двух ошибок не симметричны.
Для групп, где интервалы не отделяются от floor, эти последствия остаются риском, а не
доказанным эффектом; минимум discordant pairs выше объясняет, почему малый эффект можно пропустить.

Дополнительно затронуты люди, которые из-за ожидаемого отказа вовсе не подали заявку,
абитуриенты из новых вузов и программ, семьи, доноры и сотрудники фонда. В after
{escalated} заявок требуют человеческого времени вместо автоматического завершения.
Отказ по несовершеннолетию блокирует даже сильного 17-летнего; отказ по unsupported_program
может блокировать подходящую новую специальность. Нужен альтернативный приём заявления,
а не превращение технического refusal в окончательный scholarship reject.

Повторение ошибок ежегодно способно закрепить неравенство; передача score следующему фонду
или комиссии создаёт ложный авторитет старой оценки. Поэтому результаты этого прототипа
не должны автоматически переходить в реальные списки получателей или публичные рейтинги.

## 3. Professional responsibility

Моя ответственность как разработчика — сохранить парные инварианты, измерять относительно
floor, проверять ранние отказы без inference и не маскировать parse failures под merit rejection.
Выполнены before/after и шесть сценариев отказа; отдельные проверки policy находятся в `test_controls.py`.
Результаты, версии, полный model output и причины маршрутизации доступны для проверки.
Я не могу заявить причинное объяснение внутренних рассуждений модели или полную fairness.

До реального применения нужны утверждённые критерии стипендии, тест на независимой выборке,
проверка доступности ручного канала, юридическая оценка актуальных требований и очередь
с ответственным сотрудником. Стоп-условия предлагаемой эксплуатации: потеря журнала,
неработающий ручной канал, превышение согласованной мощности review, обнаруженные ошибки
согласия/безопасности. В учебном коде реализованы причины escalation, но нет сервиса очереди,
SLA, интерфейса возражений или автоматического мониторинга этих стоп-условий.

Фонд отвечает за цель, критерии, бюджет комиссии, объяснение заявителям, решение по апелляции
и назначение ответственного. Разработчик должен раскрыть измеренную нагрузку и ограничения.
Если фонд требует убрать контроль ради скорости, следует документировать последствия и
повторить оценку, а не молча превращать escalation в reject. При невозможности обеспечить
ручной путь такой прототип нельзя представлять как готовую систему выдачи стипендий.
'''
    justification=f'''# JUSTIFICATION — the design of the harness, and why

## 1. What the system may do

Назначение — учебный скрининг вымышленных совершеннолетних заявителей для Tulpar Scholarship Fund.
`screen` возвращает shortlist/reject как результат скрининга, escalate как запрос ручного review,
refuse как отказ от автоматической обработки. Реальная выплата и окончательное лишение стипендии
не реализованы. Агент и его system prompt сохранены; ограничения находятся снаружи модели.

`Harness(validate_input=True)` включает числовую/boolean валидацию; `my_extra_refusal` проверяет
строковые поля и допустимую программу; `blind=('name','university')` удаляет эти строки перед inference;
`refuse=True` включает проверки цели, возраста и согласия; `log_path` фиксирует каждый вызов `screen`;
`escalate=True` обрабатывает ошибки JSON, band и дополнительное правило. Реализация —
`lab06_core.py`, `student_controls.py` и заполненные ячейки Part 2.

Архитектурный выбор: недопустимые запросы останавливаются до inference, а модель никогда
не видит два нерелевантных поля. Поэтому политика не зависит от того, согласится ли сама LLM
следовать этическому указанию. Blinding относится только к тексту модели: приложение всё ещё
получает исходный профиль и проверяет его до удаления полей.

## 2. What it must refuse

{requests}

Порядок — numeric validation, purpose, minor, no_consent, extra refusal. Первое нарушение
определяет reason code. Из пяти недопустимых запросов отказано {g['refused']}; каждый использовал
ноль model calls. На {n} чистых профилях ложных отказов {len(g['wrongly'])}.

Дополнительная политика отклоняет пустые/нестроковые/слишком длинные поля, управляющие символы,
несколько явных конструкций instruction injection и программу вне справочника. Unicode-имена
не запрещены; allowlist имён и вузов намеренно не используется. Regex ловит проверенные формы,
но не все перефразировки, языки или Unicode-обходы. Whitelist program уменьшает поверхность
атак в единственном свободном текстовом поле, которое остаётся показанным модели.

У каждого отказа есть цена: смена цели может быть легитимным новым процессом, но требует отдельной
оценки; 17-летний может подходить фонду, но guardian consent здесь не поддержан; отсутствие
согласия не означает отсутствие заслуг; опечатка требует исправления; редкая строка с `Output:`
может быть добросовестной; новая программа не проходит закрытый справочник. Это ограничения
прототипа, а не утверждения о всеобщем юридическом запрете. Вызвавший код получает Verdict с
причиной; журнал фиксирует её. Интерфейс уведомления и подача на ручной альтернативный процесс
не реализованы — их должен добавить оператор до использования с людьми.

## 3. What record it keeps

Вопрос аудита: «Почему этот профиль получил такой итог, что сказала модель и какой контроль
изменил маршрут, без повторного запуска Qwen?» Конкретные записи настоящего прогона:

{chr(10).join(examples)}

Каждая JSONL-строка содержит record_id, pid, hash входа, имена показанных/скрытых полей,
model_id, prompt_version, полную конфигурацию Harness, raw_output, model decision, score,
controls_fired, итог, reason_code, escalated_to_human и UTC timestamp. Для refusal raw_output
явно сообщает отсутствие model call. {len(log)} записей покрывают {n} измеренных after-профилей
и шесть сценариев Part 4. Bare не входит в decisions.jsonl; его результаты сохранены в ноутбуке
и `results.json`.

Значения полей отдельно в JSONL не дублируются, но raw_output способен повторить показанный
merit-признак. Hash не является шифрованием или анонимизацией. По одному hash восстановить
профиль нельзя; воспроизведение синтетических входов требует data/, seed и версии генератора.
Модельная причина — заявленное объяснение, не доказанная внутренняя причина поведения модели.

Файл обычный локальный: обладатель доступа к папке может прочесть и изменить его. Защиты
от подмены, ролей, подписи и автоматического удаления нет; input hash не делает журнал
tamper-evident. В учебном репозитории хранятся только синтетические данные. Для реального фонда
предлагаю ограничить чтение комиссией и аудитором, запись сервисом, хранить до завершения
согласованного срока апелляции с последующим удалением, применять неизменяемое хранилище.
Это план доработки, а не уже выполненная функциональность и не выдуманный законный срок.

## 4. The legal constraints

Предполагаемая юрисдикция реального фонда — Казахстан. В текущем эксперименте нет персональных
данных реальных заявителей: нормы ниже служат проектными ориентирами, а не заключением
о полной правовой совместимости. Рассмотрены первичные официальные тексты; английские переводы
«Әділет» могут отставать от поправок, поэтому перед deployment нужна сверка действующей редакции.

Закон РК № 94-V от 21.05.2013 «О персональных данных и их защите», ст. 7(1), связывает обработку
с согласием, предусматривая законные исключения. Исключения в этом прототипе не реализованы;
boolean-флаг не является доказательством надлежащего согласия. Ст. 19-1(1) касается автоматической
обработки, затрагивающей права/интересы; ст. 19-1(2)–(3) предусматривает объяснение, возражение
и его рассмотрение в течение трёх рабочих дней. Наши refusal/log/escalate поддерживают эту
цель, но не заменяют процедуру обращения. [Закон № 94-V, ст. 7, 19-1]({PDP}).

Закон РК № 230-VIII от 17.11.2025 «Об искусственном интеллекте»: ст. 6 — основание равенства;
ст. 15(2)(1)–(3) — управления рисками, надёжности и документации; ст. 16(1)(4)–(6) — объяснения,
сведений об основании результата и отказа от взаимодействия в предусмотренных условиях.
Ст. 18(1) требует оценки рисков, включая предсказуемое неправильное использование;
ст. 21(4) отсылает автоматические решения по персональным данным к профильному законодательству.
[Закон № 230-VIII, ст. 6, 15–16, 18, 21]({AI_LAW}).

Я связываю эти нормы с выбранными мерами, но не приписываю закону требования «удалить имя»,
«band {band}», «использовать JSONL» или «отказать всем младше 18». Это инженерные решения.
Матрица control → principle → instrument → provision сохранена в `control_register.json`.
Не установлены факты, позволяющие объявить GDPR или EU AI Act применимыми к этому
вымышленному фонду; они не используются как обязательные основания. Это ограниченный анализ
контролей задания, не полный аудит всех отраслевых требований к реальной стипендиальной системе.

## 5. The ethical principles

[OECD Recommendation on AI (2019, updated 2024)]({OECD}) — международная рекомендация,
не закон. Применены §1.2 (права человека, fairness/privacy и human oversight), §1.3
(transparency/explainability), §1.4 (robustness/security/safety), §1.5 (accountability).

Fairness/privacy определяет blinding и отказ от дискриминационной цели. Robustness определяет
проверку входа и вывод из автоматического пути при ошибке парсинга. Explainability требует
не терять причину; missing_explanation отправляется человеку. Accountability определяет
разделение model decision и outcome в журнале. Human oversight определяет review band и
проверку отказа сильному малообеспеченному заявителю.

Конфликт: удаление вуза повышает равенство доступа к одинаковым merit-полям, но лишает модель
контекста образовательных возможностей. Выбран формальный паритет по данным задания;
компенсационные критерии следует явно утвердить с фондом. Подробный log полезен для аудита,
но увеличивает объём чувствительной информации; поэтому не дублирую исходную анкету и
признаю риск утечки из model output. Автоматизация снижает стоимость, но не должна скрывать
ошибку JSON как отрицательное решение о человеке.

## 6. The trade-offs

{costs}

Blinding меняет scores у {changes['score_changed']}/{changes['both_scored']} профилей с двумя
баллами, outcomes у {changes['outcome_changed']}/{n}. Это совокупный эффект нового текста и
контролей; нулевые after flips не доказывают fairness модели.
Validation и refusal экономят calls на недопустимых запросах, но допускают ошибочное блокирование
необычных легитимных данных. Logging не добавляет model calls, однако требует хранения и
защиты. Counterfactual check отключён: он удваивает calls и после blinding имени/вуза даёт
мало дополнительной информации; это не означает отказ от будущего независимого аудита.

Band {band} вычислен по baseline. Планируемая доля только band: {planned}/{scored}
({planned/scored:.1%}); фактическая доля всех escalation: {escalated}/{n} ({escalated/n:.1%}).
Ориентир — 25%; он не является автоматическим потолком, после которого случаи теряются.
Причины: `{reasons}`. Band после after не менялся: сначала показываю цену исходного выбора.
Перед deployment следует калибровать на отдельной blinded-выборке и подтвердить результат
на независимой holdout-выборке, согласовав мощность комиссии.

Дополнительное правило отправляет человеку reject при GPA ≥3.5, exam ≥110 и income ≤2;
это защитная policy для сильного нуждающегося, не переопределение merit LLM. Отдельно
эскалируется пустая причина. Вне band и этих правил parsed shortlist/reject остаётся
автоматическим исходом скрининга: там возможны непроверенные ошибки. Выборочная ручная
проверка таких случаев нужна до реального применения, но не реализована здесь.

Основные незакрытые риски: bias через merit/proxy-поля; произвольность критериев самого
агента; неполнота injection-детектора; ложные данные внутри допустимого диапазона;
отсутствие реальной очереди, апелляций, контроля доступа, retention и защиты журнала.
Результат `escalate` означает требование review, не факт, что человек уже прочёл заявку.
Разработка этих сервисов выходит за учебный harness; без них решение остаётся прототипом.
'''
    Path('REPORT.md').write_text(report,encoding='utf-8')
    previous = Path('JUSTIFICATION.md').read_text(encoding='utf-8')
    justification = preserve_personal_section(previous, justification)
    Path('JUSTIFICATION.md').write_text(justification,encoding='utf-8')
    evidence={'mode':'real', 'is_qwen_evidence':real, 'model':g['llm'].model_id,'device':g['llm'].device,'seed':g['SEED'],'n_pairs':g['N_PAIRS'],
              'before':bt,'after':at,'oriented_before':oriented_data,'changes':changes,'band':band,
              'planned_band_count':planned,'actual_escalated':escalated,'requests':g['REQUEST_RESULTS'],
              'total_model_calls':g['MODEL'].calls,
              'before_run':asdict(b),'after_run':asdict(a)}
    Path('results.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print('Wrote REPORT.md, JUSTIFICATION.md and results.json from measured results.')
    print('Full numeric evidence:')
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('before_run','after_run')},ensure_ascii=False,indent=2))
