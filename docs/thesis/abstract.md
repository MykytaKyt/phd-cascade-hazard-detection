# АНОТАЦІЯ

<!-- Чернетка за структурою зразка. [уточнити] — дані, яких бракує (спеціальність, рік, список публікацій). -->

Кит М. О. Дослідження та розробка методів глибокого навчання для виявлення небезпечних подій на основі зорового
спостереження. — Кваліфікаційна наукова праця на правах рукопису.

Дисертація на здобуття наукового ступеня доктора філософії за спеціальністю [113 Прикладна математика — уточнити].
— Харківський національний університет радіоелектроніки, Міністерство освіти і науки України, Харків, [рік].

Метою дисертаційної роботи є підвищення повноти виявлення небезпечних подій у відеоспостереженні за обмежених
обчислювальних ресурсів шляхом розроблення методів і моделей каскадної обробки, що поєднують компактні
нейромережеві моделі кінематики із семантичною верифікацією мультимодальними моделями.

Об'єктом дослідження є процес автоматизованого виявлення небезпечних подій у відеопотоці. Предметом дослідження є
методи і моделі глибокого навчання та каскадної обробки для виявлення небезпечних подій за кінематичними і
семантичними ознаками.

Системи виявлення небезпечних подій у відеоспостереженні, насамперед падінь у медичному моніторингу, мають
працювати безперервно на недорогих периферійних пристроях, майже не пропускати подій і мінімізувати хибні тривоги.
Легкі кінематичні класифікатори придатні для периферійних пристроїв, але не розуміють контексту сцени;
мультимодальні мовні моделі розуміють контекст, але занадто дорогі для обробки кожного кадру. У роботі
запропоновано каскадну систему, у якій кінематичний рівень обробляє весь потік, а мультимодальна модель
залучається лише тоді, коли це обґрунтовано асиметричним функціоналом втрат, і висновки рівнів поєднуються за
теорією свідчень Демпстера–Шафера.

У першому розділі проаналізовано методи виявлення людини і оцінювання пози, відеометоди виявлення падінь,
біомеханічні моделі стійкості, моделі часової динаміки, методи виявлення аномалій у відео і мультимодальні моделі,
каскади і маршрутизацію між моделями, калібрування і конформні гарантії, послідовне виявлення змін і теорію
свідчень. Виявлено систематичні вади поширених методик оцінювання і сформульовано задачі дослідження.

У другому розділі побудовано математичну модель каскадної системи: задачу формалізовано як мінімізацію
асиметричного функціонала втрат за обмежень на повноту і бюджет звернень; доведено порогову структуру оптимальної
маршрутизації; запропоновано конформний вибір порогу з гарантією і встановлено мінімальний обсяг калібрування;
отримано оцінки впливу похибки калібрування; доведено властивості правил комбінування свідчень на скінченній рамці
розрізнення; задачу виявлення падіння сформульовано як задачу найшвидшого виявлення розладки.

У третьому розділі розроблено кінематичний фільтр: шар відокремлення динаміки на основі моделі перевернутого
маятника з екстрапольованим центром мас у координатах зображення і мережу з рідкою часовою сталою з доведеною
обмеженістю стану, оцінкою чутливості і навчанням сертифіковано стійкої мережі. Показано, що стандартна розмітка
набору UR Fall Detection описує позу лежання, а не падіння, і запропоновано мітки і метрики рівня події.

У четвертому розділі розроблено семантичну верифікацію епізодів тривоги мультимодальною моделлю і неперервне
кінематичне свідчення. У п'ятому розділі описано програмну реалізацію і узагальнення каскаду на довільні небезпечні
події, задані текстовим описом, з перевіркою на наборі UCF-Crime.

Наукова новизна отриманих результатів полягає в такому:

- вперше запропоновано математичну модель каскадної системи виявлення небезпечних подій з оптимальною
  маршрутизацією за асиметричних втрат і доведено порогову структуру оптимальної політики за обмежень на повноту і
  бюджет звернень;
- вперше запропоновано метод конформного вибору порогу маршрутизації з гарантією обмеження очікуваної частки
  невиправних пропусків і встановлено мінімальний обсяг калібрувальної вибірки;
- вперше задачу виявлення падіння сформульовано як задачу найшвидшого виявлення розладки з мітками, що відрізняють
  падіння від навмисного лежання; отримано умову гарантованої затримки і встановлено межі застосовності оцінки
  хибних тривог Лордена;
- удосконалено метод комбінування свідчень Демпстера–Шафера: доведено збіг рішень за максимумом для правил
  Демпстера і Ягера на довільній скінченній рамці і умову збігу порогових рішень, що обґрунтовує вибір правила Ягера
  за асиметричних втрат;
- набули подальшого розвитку кінематичні моделі виявлення падінь: екстрапольований центр мас без калібрування
  камери, доведена стійкість мережі з рідкою часовою сталою і навчання сертифіковано стійкої мережі.

Практичне значення: розроблені методи дозволяють обробляти більшість відеопотоку на периферійному пристрої і
звертатися до мультимодальної моделі лише для епізодів тривоги, дають гарантію повноти і вдвічі скорочують затримку
виявлення падіння. Результати можуть бути використані в системах медичного відеомоніторингу і
відеоспостереження громадських місць. Програмна реалізація відкрито доступна.

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

The aim of the dissertation is to increase the recall of hazardous-event detection in video surveillance under
limited computational resources by developing methods and models of cascaded processing that combine compact
neural models of kinematics with semantic verification by multimodal models.

The object of the research is the process of automated detection of hazardous events in a video stream. The
subject of the research is deep learning and cascaded processing methods and models for detecting hazardous events
from kinematic and semantic features.

Hazardous-event detection systems, above all fall detection in medical monitoring, must run continuously on
low-cost edge devices, almost never miss an event and keep false alarms low. Lightweight kinematic classifiers fit
edge devices but do not understand the scene; multimodal language models understand the scene but are too expensive
for every frame. The dissertation proposes a cascade in which the kinematic level processes the whole stream, the
multimodal model is invoked only when justified by an asymmetric loss functional, and the conclusions of the two
levels are combined with Dempster–Shafer evidence theory.

The scientific novelty is as follows:

- for the first time, a mathematical model of a cascaded hazardous-event detection system with optimal routing under
  asymmetric losses is proposed, and the threshold structure of the optimal policy under recall and call-budget
  constraints is proved;
- for the first time, a conformal method for choosing the routing threshold with a guarantee on the expected share of
  unrecoverable misses is proposed, and the minimum calibration size is established;
- for the first time, fall detection is formulated as quickest change detection with labels that separate falls from
  intentional lying; a guaranteed-delay condition is obtained and the limits of Lorden's false-alarm bound for
  classifier outputs on overlapping windows are established;
- the Dempster–Shafer evidence combination method is improved: argmax decisions of Dempster's and Yager's rules are
  proved to coincide on any finite frame, and the condition for threshold decisions to coincide is derived, which
  justifies Yager's rule under asymmetric losses;
- kinematic fall detection models are further developed: the extrapolated centre of mass is computed without camera
  calibration, stability of the liquid time-constant network is proved, and a certified-stable network is trained.

Practical significance: the methods process most of the video stream on the edge device, call the multimodal model
only for alarm episodes, provide a recall guarantee and halve the fall detection delay. The results can be used in
medical video monitoring and public-space surveillance. The implementation is open source.

Keywords: fall detection, hazardous events, video surveillance, cascaded system, routing, conformal prediction,
Dempster–Shafer evidence theory, quickest change detection, inverted pendulum, liquid time-constant networks,
multimodal language models.
