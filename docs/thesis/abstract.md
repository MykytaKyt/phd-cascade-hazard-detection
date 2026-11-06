# АНОТАЦІЯ

<!-- [уточнити] — дані, яких бракує: спеціальність, рік, вихідні дані публікацій (docs/publications.md). -->

Кит М. О. Дослідження та розробка методів глибокого навчання для виявлення небезпечних подій на основі зорового
спостереження. — Кваліфікаційна наукова праця на правах рукопису.

Дисертація на здобуття наукового ступеня доктора філософії за спеціальністю [113 Прикладна математика — уточнити].
— Харківський національний університет радіоелектроніки, Міністерство освіти і науки України, Харків, [рік].

Дисертаційна робота присвячена розв'язанню актуальної наукової задачі розроблення математичних моделей і методів
прийняття рішень про небезпечну подію за послідовністю зорових спостережень з використанням джерел інформації
різної вартості — компактних нейромережевих моделей і мультимодальних моделей глибокого навчання — з гарантіями
повноти виявлення за обмежених обчислювальних ресурсів.

Метою дисертаційної роботи є розроблення математичних моделей і методів прийняття рішень про небезпечну подію
за послідовністю зорових спостережень, які поєднують джерела інформації різної вартості і забезпечують
гарантовану повноту виявлення за обмежених обчислювальних ресурсів.

Об'єктом дослідження є процес прийняття рішень про небезпечну подію за послідовністю зорових спостережень в
умовах невизначеності. Предметом дослідження є математичні моделі і методи маршрутизації, комбінування свідчень і
послідовного виявлення змін для джерел інформації різної вартості, зокрема моделей глибокого навчання.

Виявлення небезпечної події за відеоспостереженням розглядається як задача послідовного прийняття рішень в умовах
невизначеності з асиметричними вартостями помилок і обмеженим обчислювальним ресурсом. Відомі математичні
моделі — оцінювання пози, біомеханічної стійкості, часової динаміки, калібрування, відмови від рішення, виявлення
розладки, комбінування свідчень — розв'язують лише окремі частини цієї задачі і не дають гарантій повноти. У роботі
задачу сформульовано як вибір політики звернень до двох джерел інформації різної вартості: дешевого, що обробляє
кожне спостереження, і дорогого (мультимодальної моделі, VLM), що інтерпретує контекст сцени; доведено, що
оптимальна політика має порогову структуру, з якої випливає каскадна організація системи, а свідчення джерел
поєднуються за теорією Демпстера–Шафера.

У першому розділі проаналізовано методи виявлення людини і оцінювання пози, відеометоди виявлення падінь,
біомеханічні моделі стійкості, моделі часової динаміки, методи виявлення аномалій у відео, мультимодальні моделі,
каскади і маршрутизацію між моделями, калібрування і конформні гарантії, послідовне виявлення змін і теорію
свідчень. Виявлено систематичні вади поширених методик оцінювання (вибір порогу за тестовими мітками, покадрові
метрики, розмітка пози лежання замість падіння) і сформульовано задачі дослідження.

У другому розділі побудовано математичну модель прийняття рішень за двома джерелами інформації різної вартості.
Задачу формалізовано як мінімізацію
асиметричного функціонала втрат за обмежень на повноту і бюджет звернень; доведено порогову структуру оптимальної
маршрутизації, у якій обмеження на повноту визначає лише нижній поріг. Запропоновано вибір порогу контролем
конформного ризику з одиницею обмінюваності «запис» і доведено гарантію повноти в середньому і з високою
ймовірністю; встановлено мінімальний обсяг калібрувальної вибірки. Отримано оцінки впливу похибки калібрування на
межу повноти. Для правил Демпстера і Ягера на довільній скінченній рамці доведено збіг рішень за максимумом
пігністичної ймовірності і умову збігу порогових рішень β = |B|/n. Отримано умову доцільності каскаду. Задачу
виявлення падіння сформульовано як задачу найшвидшого виявлення розладки на виході класифікатора; отримано умову
гарантованої затримки і показано, що оцінка хибних тривог Лордена не переноситься на вихід класифікатора на
перекритих вікнах. Показано, що каскадна структура системи випливає з отриманих результатів.

У третьому розділі розроблено кінематичний фільтр: шар відокремлення динаміки на основі моделі перевернутого
маятника з екстрапольованим центром мас у координатах зображення і мережу з рідкою часовою сталою (1 753 параметри)
з доведеною обмеженістю стану, оцінкою чутливості і навчанням сертифіковано стійкої мережі, що зменшує чутливість у
5.5 раза без статистично значущої втрати якості. Показано, що стандартна бінаризація розмітки набору UR Fall
Detection описує позу лежання (62 % позитивних кадрів — навмисне лягання), запропоновано мітки «падіння», які
вдвічі скорочують затримку виявлення, і показано, що висновок про достатність ознак обмежувальної рамки є
наслідком розмітки лежання.

У четвертому розділі розроблено семантичну верифікацію епізодів тривоги мультимодальною моделлю і неперервне
кінематичне свідчення. Верифікація потребує ≈ 37 звернень до VLM на прогін URFD замість ≈ 3 000 за покадрової
маршрутизації. На чотирьох VLM (Gemma 4 E4B і 31B, Qwen3-VL-8B і 32B) за однакових запитів показано, що
розрізнювальна здатність семантичного рівня зростає з розміром моделі (ROC AUC 0.49–0.92 на кліпах) і практично
відсутня на окремих кадрах (0.47–0.57). З Qwen3-VL-32B каскад зменшує кількість хибних тривог у 2.5 раза
(12.6 → 5.0 на прогін) ціною 0.2 пропущеного падіння і вигідний, якщо пропуск дорожчий за хибну тривогу не більше
ніж у 38 разів; з компактною Gemma 4 E4B — лише за 1.6. З умови доцільності отримано вимогу до робочої точки
верифікатора, яка дає змогу оцінити придатність VLM до розгортання.

У п'ятому розділі описано програмну реалізацію і узагальнення каскаду на довільні небезпечні події, задані текстовим
описом. На наборі UCF-Crime (13 класів подій) перший рівень — модель CLIP без донавчання (ROC AUC 73.6–75.6 %);
конформний вибір порогу виконує гарантію повноти в усіх розбиттях, тоді як емпіричний поріг порушує її в 1–2 з 5;
семантичний рівень підвищує точність тривог з 0.14 до 0.51 і зменшує частку нормальних відео з тривогою з 46 % до
3 %; на 14-класовій рамці підтверджено умову збігу рішень правил комбінування.

Наукова новизна отриманих результатів полягає в такому:

- вперше побудовано математичну модель прийняття рішення про небезпечну подію за двома джерелами інформації
  різної вартості як задачу мінімізації асиметричного функціонала втрат за обмежень на повноту і бюджет звернень і
  доведено двопорогову структуру оптимальної політики, з якої випливає каскадна структура системи;
- вперше запропоновано метод вибору порогу маршрутизації контролем конформного ризику з гарантією обмеження
  частки невиправних пропусків (у середньому і з високою ймовірністю) і встановлено мінімальний обсяг
  калібрувальної вибірки;
- вперше отримано умову доцільності звернень до дорогого джерела у вигляді вимоги до його робочої точки;
- вперше задачу виявлення падіння сформульовано як задачу найшвидшого виявлення розладки на виході
  класифікатора з мітками, що відрізняють падіння від навмисного лежання; отримано умову гарантованої затримки і
  встановлено межі застосовності оцінки хибних тривог Лордена;
- удосконалено метод комбінування свідчень Демпстера–Шафера: запропоновано неперервне свідчення
  класифікатора, доведено збіг рішень за максимумом для правил Демпстера і Ягера на довільній скінченній рамці і
  умову збігу порогових рішень;
- набули подальшого розвитку математичні моделі кінематики падіння: екстрапольований центр мас без
  калібрування камери, доведена стійкість мережі з рідкою часовою сталою і навчання сертифіковано стійкої мережі.

Практичне значення: розроблені методи дозволяють обробляти весь відеопотік на периферійному пристрої моделлю з
кількома тисячами параметрів і звертатися до мультимодальної моделі лише для епізодів тривоги; дають гарантію
повноти, яку емпіричний вибір порогу порушує; вдвічі скорочують затримку виявлення падіння; зменшують кількість
хибних тривог у 2.5 раза за достатньо сильної мультимодальної моделі. Результати можуть бути використані в
системах медичного відеомоніторингу, відеоспостереження громадських місць і пошуку у відеоархівах за текстовим
описом. Програмна реалізація відкрито доступна і забезпечує відтворюваність усіх числових результатів.

Ключові слова: прийняття рішень в умовах невизначеності, асиметричні втрати, виявлення падінь, небезпечні події, відеоспостереження, каскадна система, маршрутизація,
конформне передбачення, теорія свідчень Демпстера–Шафера, найшвидше виявлення розладки, перевернутий маятник,
мережі з рідкою часовою сталою, мультимодальні мовні моделі.

## Список публікацій здобувача

**Наукові праці, в яких опубліковані основні наукові результати дисертації**

1. [Кит М. О., Єсілевський В. С. Системний аналіз і математична модель каскадного виявлення падінь людини з
   ентропійною маршрутизацією та комбінуванням свідчень Демпстера–Шафера. Видання, рік, том, №, сторінки, DOI.]
2. [Кит М. О., Єсілевський В. С. Кінематичний фільтр виявлення падінь людини на основі біомеханічної моделі
   перевернутого маятника та мережі з рідкою часовою сталою. Видання, рік, №, сторінки, DOI.]
3. [Kyt M., Yesilevskyi V. A Cascaded Method for Detecting Spatio-Temporal Anomalies of Human Posture in Medical Video
   Surveillance. International Journal of Online and Biomedical Engineering (iJOE), рік, том, №, сторінки, DOI.]
4. [Kyt M., Yesilevskyi V. Mobile video platform for diagnosing movement anomalies. Видання, рік, сторінки.]

**Наукові праці, які засвідчують апробацію матеріалів дисертації**

5. [Тези: Мобільна відеоплатформа для діагностики рухових аномалій.]
6. [Тези ІКТК-2025: Концептуальна архітектура мобільної відеоплатформи для діагностики рухових аномалій.]
7. [Тези 2025: Математичні методи та алгоритми відеоаналізу рухових порушень.]
8. [Тези 2026: Пошук небезпечних подій у відеоархівах спостереження за текстовим описом на основі моделей
   глибокого навчання.]

---

# SUMMARY

Kyt M. O. Research and development of deep learning methods for detecting hazardous events based on visual
surveillance. — Qualification scientific work, manuscript.

Dissertation for the degree of Doctor of Philosophy in specialty [113 Applied Mathematics — to be confirmed].
— Kharkiv National University of Radio Electronics, Ministry of Education and Science of Ukraine, Kharkiv, [year].

The dissertation solves the scientific problem of developing mathematical models and methods of decision making
about a hazardous event from a sequence of visual observations using information sources of different cost —
compact neural models and multimodal deep learning models — with guarantees on detection recall under limited
computational resources.

The aim of the dissertation is to develop mathematical models and methods of decision making about a hazardous
event from a sequence of visual observations that combine information sources of different cost and guarantee the
detection recall under limited computational resources.

The object of the research is the process of decision making about a hazardous event from a sequence of visual
observations under uncertainty. The subject of the research is mathematical models and methods of routing, evidence
combination and sequential change detection for information sources of different cost, including deep learning
models.

Detection of a hazardous event from video is treated as sequential decision making under uncertainty with
asymmetric error costs and a limited computational budget. Known mathematical models — pose estimation,
biomechanical stability, temporal dynamics, calibration, rejection, change detection, evidence combination — solve
only parts of this problem and give no recall guarantees. The problem is formulated as the choice of a policy of
calls to two information sources of different cost: a cheap one that processes every observation and an expensive
one (a vision-language model, VLM) that interprets the scene. The optimal policy is proved to have a threshold
structure, from which the cascaded organisation of the system follows; the evidence of the two sources is combined
with Dempster–Shafer theory.

The first chapter reviews person detection and pose estimation, video fall detection, biomechanical stability
models, temporal models, video anomaly detection, multimodal models, cascades and model routing, calibration and
conformal guarantees, sequential change detection and evidence theory. Systematic flaws of common evaluation
practice are identified: thresholds chosen on test labels, frame-level metrics, and labels of the lying pose instead
of the fall.

The second chapter builds a mathematical model of decision making with two information sources of different cost. The problem is formalised as minimisation of an
asymmetric loss under recall and call-budget constraints; the optimal routing is proved to have a threshold
structure in which the recall constraint fixes only the lower threshold. The routing threshold is chosen by
conformal risk control with the recording as the exchangeable unit, with a guarantee in expectation and with high
probability, and the minimum calibration size is established. Bounds on the effect of calibration error are derived.
For Dempster's and Yager's rules on any finite frame, argmax decisions of the pignistic probability are proved to
coincide, and threshold decisions to coincide exactly at β = |B|/n. A cascade feasibility condition is obtained. Fall
detection is formulated as quickest change detection on the classifier output; a guaranteed-delay condition is
derived, and Lorden's false-alarm bound is shown not to carry over to classifier outputs on overlapping windows.

The third chapter develops the kinematic filter: a dynamics-decoupling layer based on the inverted pendulum with
the extrapolated centre of mass in image coordinates, and a liquid time-constant network (1,753 parameters) with a
proved state bound, a sensitivity bound and certified-stable training that reduces the sensitivity 5.5-fold without
a significant loss of quality. The standard binarisation of the UR Fall Detection labels is shown to mark the lying
pose (62 % of positive frames are intentional lying); fall labels halve the detection delay, and the finding that
bounding-box features suffice is shown to be an artefact of the lying labels.

The fourth chapter develops semantic verification of alarm episodes by a VLM and a continuous kinematic evidence.
Verification needs about 37 VLM calls per URFD run instead of about 3,000 for frame-level routing. With four VLMs
(Gemma 4 E4B and 31B, Qwen3-VL-8B and 32B) and identical requests, the discriminative power of the semantic level
grows with the model size (ROC AUC 0.49–0.92 on clips) and is almost absent on single frames (0.47–0.57). With
Qwen3-VL-32B the cascade reduces false alarms 2.5-fold (12.6 → 5.0 per run) at the cost of 0.2 missed falls and pays
off whenever a miss costs at most 38 false alarms; with the compact Gemma 4 E4B, only up to 1.6. The feasibility
condition yields a requirement on the operating point of the verifier that allows assessing a VLM before deployment.

The fifth chapter describes the software and generalises the cascade to arbitrary hazardous events specified by
text. On UCF-Crime (13 event classes) the first level is zero-shot CLIP (ROC AUC 73.6–75.6 %); conformal threshold
choice meets the recall guarantee in all splits, whereas the empirical threshold violates it in 1–2 of 5; the
semantic level raises alarm precision from 0.14 to 0.51 and reduces the share of normal videos with an alarm from
46 % to 3 %; the coincidence condition for the combination rules is confirmed on the 14-class frame.

The scientific novelty is as follows:

- for the first time, a mathematical model of decision making about a hazardous event with two information sources
  of different cost is built as minimisation of an asymmetric loss under recall and call-budget constraints, and the
  optimal call policy is proved to have a two-threshold structure, from which the cascaded structure of the system
  follows;
- for the first time, the routing threshold is chosen by conformal risk control with a guarantee on the share of
  unrecoverable misses (in expectation and with high probability), and the minimum calibration size is established;
- for the first time, the condition under which calls to the expensive source pay off is stated as a requirement on
  its operating point;
- for the first time, fall detection is formulated as quickest change detection on the classifier output with labels
  that separate falls from intentional lying; a guaranteed-delay condition is obtained and the limits of Lorden's
  false-alarm bound are established;
- the Dempster–Shafer combination method is improved: a continuous classifier evidence is proposed, argmax decisions
  of Dempster's and Yager's rules are proved to coincide on any finite frame, and the condition for threshold
  decisions to coincide is derived;
- mathematical models of fall kinematics are further developed: the extrapolated centre of mass without camera
  calibration, stability of the liquid time-constant network and certified-stable training.

Practical significance: the methods process the whole video stream on the edge device with a model of a few
thousand parameters and call the multimodal model only for alarm episodes; they provide a recall guarantee that
empirical thresholds violate, halve the fall detection delay and reduce false alarms 2.5-fold with a sufficiently
strong multimodal model. The results can be used in medical video monitoring, public-space surveillance and
text-based search in video archives. The implementation is open source and reproduces every number of the work.

Keywords: decision making under uncertainty, asymmetric losses, fall detection, hazardous events, video surveillance, cascaded system, routing, conformal prediction,
Dempster–Shafer evidence theory, quickest change detection, inverted pendulum, liquid time-constant networks,
vision-language models.
