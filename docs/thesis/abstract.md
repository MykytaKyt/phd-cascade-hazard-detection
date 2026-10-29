# АНОТАЦІЯ

<!-- [уточнити] — дані, яких бракує: спеціальність, рік, вихідні дані публікацій (docs/publications.md). -->

Кит М. О. Дослідження та розробка методів глибокого навчання для виявлення небезпечних подій на основі зорового
спостереження. — Кваліфікаційна наукова праця на правах рукопису.

Дисертація на здобуття наукового ступеня доктора філософії за спеціальністю [113 Прикладна математика — уточнити].
— Харківський національний університет радіоелектроніки, Міністерство освіти і науки України, Харків, [рік].

Дисертаційна робота присвячена розв'язанню актуальної наукової задачі розроблення математичних моделей і методів
каскадного виявлення небезпечних подій у відеоспостереженні з формально обґрунтованою маршрутизацією між
компактною кінематичною моделлю і мультимодальною моделлю та комбінуванням їхніх свідчень.

Метою дисертаційної роботи є підвищення повноти виявлення небезпечних подій у відеоспостереженні за обмежених
обчислювальних ресурсів шляхом розроблення методів і моделей каскадної обробки, що поєднують компактні
нейромережеві моделі кінематики із семантичною верифікацією мультимодальними моделями.

Об'єктом дослідження є процес автоматизованого виявлення небезпечних подій у відеопотоці. Предметом дослідження є
методи і моделі глибокого навчання та каскадної обробки для виявлення небезпечних подій за кінематичними і
семантичними ознаками.

Системи виявлення небезпечних подій, насамперед падінь у медичному моніторингу, мають працювати безперервно на
недорогих периферійних пристроях, майже не пропускати подій і мінімізувати хибні тривоги. Легкі кінематичні
класифікатори придатні для периферійних пристроїв, але не розуміють контексту сцени; мультимодальні мовні моделі
(VLM) розуміють контекст, але занадто дорогі для обробки кожного кадру. У роботі запропоновано каскадну систему, у
якій кінематичний рівень обробляє весь потік, мультимодальна модель залучається лише тоді, коли це обґрунтовано
асиметричним функціоналом втрат, а висновки рівнів поєднуються за теорією свідчень Демпстера–Шафера.

У першому розділі проаналізовано методи виявлення людини і оцінювання пози, відеометоди виявлення падінь,
біомеханічні моделі стійкості, моделі часової динаміки, методи виявлення аномалій у відео, мультимодальні моделі,
каскади і маршрутизацію між моделями, калібрування і конформні гарантії, послідовне виявлення змін і теорію
свідчень. Виявлено систематичні вади поширених методик оцінювання (вибір порогу за тестовими мітками, покадрові
метрики, розмітка пози лежання замість падіння) і сформульовано задачі дослідження.

У другому розділі побудовано математичну модель каскадної системи. Задачу формалізовано як мінімізацію
асиметричного функціонала втрат за обмежень на повноту і бюджет звернень; доведено порогову структуру оптимальної
маршрутизації, у якій обмеження на повноту визначає лише нижній поріг. Запропоновано вибір порогу контролем
конформного ризику з одиницею обмінюваності «запис» і доведено гарантію повноти в середньому і з високою
ймовірністю; встановлено мінімальний обсяг калібрувальної вибірки. Отримано оцінки впливу похибки калібрування на
межу повноти. Для правил Демпстера і Ягера на довільній скінченній рамці доведено збіг рішень за максимумом
пігністичної ймовірності і умову збігу порогових рішень β = |B|/n. Отримано умову доцільності каскаду. Задачу
виявлення падіння сформульовано як задачу найшвидшого виявлення розладки на виході класифікатора; отримано умову
гарантованої затримки і показано, що оцінка хибних тривог Лордена не переноситься на вихід класифікатора на
перекритих вікнах.

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

- вперше запропоновано математичну модель каскадної системи виявлення небезпечних подій з оптимальною
  маршрутизацією за асиметричних втрат і доведено порогову структуру оптимальної політики за обмежень на повноту і
  бюджет звернень;
- вперше запропоновано метод конформного вибору порогу маршрутизації з гарантією обмеження очікуваної частки
  невиправних пропусків, у тому числі з високою ймовірністю, і встановлено мінімальний обсяг калібрувальної
  вибірки;
- вперше задачу виявлення падіння сформульовано як задачу найшвидшого виявлення розладки з мітками, що відрізняють
  падіння від навмисного лежання; отримано умову гарантованої затримки і встановлено межі застосовності оцінки
  хибних тривог Лордена;
- удосконалено метод комбінування свідчень Демпстера–Шафера: запропоновано неперервне кінематичне свідчення,
  доведено збіг рішень за максимумом для правил Демпстера і Ягера на довільній скінченній рамці і умову збігу
  порогових рішень;
- вперше отримано умову доцільності каскаду у вигляді вимоги до робочої точки семантичного верифікатора і
  експериментально встановлено залежність внеску семантичного рівня від розміру мультимодальної моделі;
- набули подальшого розвитку кінематичні моделі виявлення падінь: екстрапольований центр мас без калібрування
  камери, доведена стійкість мережі з рідкою часовою сталою і навчання сертифіковано стійкої мережі.

Практичне значення: розроблені методи дозволяють обробляти весь відеопотік на периферійному пристрої моделлю з
кількома тисячами параметрів і звертатися до мультимодальної моделі лише для епізодів тривоги; дають гарантію
повноти, яку емпіричний вибір порогу порушує; вдвічі скорочують затримку виявлення падіння; зменшують кількість
хибних тривог у 2.5 раза за достатньо сильної мультимодальної моделі. Результати можуть бути використані в
системах медичного відеомоніторингу, відеоспостереження громадських місць і пошуку у відеоархівах за текстовим
описом. Програмна реалізація відкрито доступна і забезпечує відтворюваність усіх числових результатів.

Ключові слова: виявлення падінь, небезпечні події, відеоспостереження, каскадна система, маршрутизація,
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

The dissertation solves the scientific problem of developing mathematical models and methods for cascaded
detection of hazardous events in video surveillance, with formally justified routing between a compact kinematic
model and a multimodal model and with the combination of their evidence.

The aim of the dissertation is to increase the recall of hazardous-event detection in video surveillance under
limited computational resources by developing methods and models of cascaded processing that combine compact
neural models of kinematics with semantic verification by multimodal models.

The object of the research is the process of automated detection of hazardous events in a video stream. The
subject of the research is deep learning and cascaded processing methods and models for detecting hazardous events
from kinematic and semantic features.

Hazardous-event detection systems, above all fall detection in medical monitoring, must run continuously on
low-cost edge devices, almost never miss an event and keep false alarms low. Lightweight kinematic classifiers fit
edge devices but do not understand the scene; vision-language models (VLMs) understand the scene but are too
expensive for every frame. The dissertation proposes a cascade in which the kinematic level processes the whole
stream, the multimodal model is invoked only when justified by an asymmetric loss functional, and the conclusions
of the two levels are combined with Dempster–Shafer evidence theory.

The first chapter reviews person detection and pose estimation, video fall detection, biomechanical stability
models, temporal models, video anomaly detection, multimodal models, cascades and model routing, calibration and
conformal guarantees, sequential change detection and evidence theory. Systematic flaws of common evaluation
practice are identified: thresholds chosen on test labels, frame-level metrics, and labels of the lying pose instead
of the fall.

The second chapter builds a mathematical model of the cascade. The problem is formalised as minimisation of an
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

- for the first time, a mathematical model of a cascaded hazardous-event detection system with optimal routing under
  asymmetric losses is proposed, and the threshold structure of the optimal policy under recall and call-budget
  constraints is proved;
- for the first time, a conformal method for choosing the routing threshold with a guarantee on the expected share of
  unrecoverable misses, including a high-probability guarantee, is proposed, and the minimum calibration size is
  established;
- for the first time, fall detection is formulated as quickest change detection with labels that separate falls from
  intentional lying; a guaranteed-delay condition is obtained and the limits of Lorden's false-alarm bound are
  established;
- the Dempster–Shafer evidence combination method is improved: a continuous kinematic evidence is proposed, argmax
  decisions of Dempster's and Yager's rules are proved to coincide on any finite frame, and the condition for
  threshold decisions to coincide is derived;
- for the first time, the cascade feasibility condition is stated as a requirement on the operating point of the
  semantic verifier, and the dependence of the contribution of the semantic level on the VLM size is established
  experimentally;
- kinematic fall detection models are further developed: the extrapolated centre of mass is computed without camera
  calibration, stability of the liquid time-constant network is proved, and a certified-stable network is trained.

Practical significance: the methods process the whole video stream on the edge device with a model of a few
thousand parameters and call the multimodal model only for alarm episodes; they provide a recall guarantee that
empirical thresholds violate, halve the fall detection delay and reduce false alarms 2.5-fold with a sufficiently
strong multimodal model. The results can be used in medical video monitoring, public-space surveillance and
text-based search in video archives. The implementation is open source and reproduces every number of the work.

Keywords: fall detection, hazardous events, video surveillance, cascaded system, routing, conformal prediction,
Dempster–Shafer evidence theory, quickest change detection, inverted pendulum, liquid time-constant networks,
vision-language models.
