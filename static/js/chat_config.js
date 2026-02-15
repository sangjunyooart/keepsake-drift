// static/js/chat_config.js
// 긴 텍스트/매핑/기본값은 전부 여기로.
// runtime은 동작만 담당.

window.KD_CHAT_CONFIG = {
  // 아라빅 기능 켜기/끄기 (여기 한 줄로 제어)
  ENABLE_ARABIC: true,

  // 폴링 주기 (ms)
  STATE_POLL_MS: 3000,

  // localStorage keys
  STORAGE_LANG_KEY: "kd_lang",
  STORAGE_CHAT_KEY: "kd_chat_histories_v1",
  STORAGE_SESSION_PREFIX: "kd_session_",

  // 백엔드 TEMPORALITIES와 반드시 일치
  VALID_PERSONAS: ["human", "liminal", "environment", "digital", "infrastructure", "more_than_human"],

  // UI 표시용 라벨 (기존 텍스트 유지)
  uiText: {
    en: {
      langButton: "عرب",
      personaLabel: {
        liminal: "Liminal-time",
        human: "Human-time",
        environment: "Environmental-time",
        digital: "Digital-time",
        infrastructure: "Infrastructure-time",
        more_than_human: "More-than-human-time"
      },
      placeholder: "enter a memory fragment...",
      you: "Me",
      aiByPersona: {
        liminal: "Liminal-time",
        human: "Human-time",
        environment: "Environmental-time",
        digital: "Digital-time",
        infrastructure: "Infrastructure-time",
        more_than_human: "More-than-human-time"
      },
      listening: "Listening to your memory...",
      errorPrefix: "Error: ",
      driftBtnPrefix: "Drift",
      keepsakeBtn: "Keepsake",

      drift: {
        liminal:
`Tonight the hills felt slightly different. The usual quiet between the evening sounds still settled over the neighborhood, but within it I sensed a faint shimmer moving through the air, as if the dusk light had brushed the streets with a soft brightness.`,
        human:
`The day only became clear when it returned again in small details, the way a breath repeats itself without asking.`,
        environment:
`The air changed first, and only later did I notice how much of the memory had moved with it.`,
        digital:
`Between signal and surface, the image kept arriving slightly late, as if it needed time to remember itself.`,
        infrastructure:
`I followed the line where the road pretended to be permanent, and felt the soft shake of everything underneath.`,
        more_than_human:
`Something smaller than my attention carried the scene, turning it over and over until it shone.`
      },

      keepsake: {
        liminal:
`I stood at the edge of something — not quite night, not quite morning. The stillness held, but something underneath it had started to thin.`,
        human:
`I felt the day return in small details, the way a breath repeats itself without asking.`,
        environment:
`I noticed the air shift first. Only later did I feel how much had moved with it.`,
        digital:
`I kept receiving the image slightly late, as if it needed time to remember itself before reaching me.`,
        infrastructure:
`I followed the line where the road pretended to be permanent, and felt the soft shake underneath.`,
        more_than_human:
`Something smaller than my attention carried the scene, turning it over until it shone.`
      }
    },

    ar: {
      langButton: "EN",
      personaLabel: {
        liminal: "الزمن البيني",
        human: "الزمن الإنساني",
        environment: "الزمن البيئي",
        digital: "الزمن الرقمي",
        infrastructure: "زمن البنية التحتية",
        more_than_human: "زمن ما-بعد-الإنسان"
      },
      placeholder: "أدخل جزءًا من الذاكرة...",
      you: "أنا",
      aiByPersona: {
        liminal: "الزمن البيني",
        human: "الزمن الإنساني",
        environment: "الزمن البيئي",
        digital: "الزمن الرقمي",
        infrastructure: "زمن البنية التحتية",
        more_than_human: "زمن ما-بعد-الإنسان"
      },
      listening: "أستمع إلى ذاكرتك...",
      errorPrefix: "خطأ: ",
      driftBtnPrefix: "انجراف",
      keepsakeBtn: "ذاكرة",

      drift: {
        liminal:
`هذه الليلة بدت التلال مختلفة قليلاً. الهدوء المعتاد بين أصوات المساء ظل مستقرًا فوق الحي، لكنني شعرت داخله بوميض خافت يتحرك في الهواء، كأن ضوء الغسق مر على الشوارع بسطوع ناعم. لم يكن ضجيجًا أو حركة في الأسفل، بل تمددًا هادئًا، كأن التوقف نفسه يحمل توقعًا لطيفًا.

بدت اللمعة على الجدران أكثر طبقات من المعتاد، كما تتبدل الانعكاسات حين يتغير الموسم. شعرت الألوان أبرد وأدفأ في الوقت نفسه، صدى مكتوم لشيء يتشكل خارج النظر، لا يصل إلا إلى أطراف السكون.`,
        human:
`لا يتضح اليوم إلا عندما يعود في تفاصيل صغيرة، كأن النفس يكرر نفسه دون سؤال.`,
        environment:
`تغير الهواء أولاً، وبعدها فقط لاحظت كم تحركت الذاكرة معه.`,
        digital:
`بين الإشارة والسطح، كانت الصورة تصل متأخرة قليلاً، كأنها تحتاج وقتًا لتتذكر نفسها.`,
        infrastructure:
`اتبعت الخط الذي يتظاهر فيه الطريق بالثبات، وشعرت بالارتجاف الناعم لكل ما تحته.`,
        more_than_human:
`شيء أصغر من انتباهي حمل المشهد، يقلبه مرارًا حتى يلمع.`
      },

      keepsake: {
        liminal:
`وقفتُ عند حافة شيء ما — ليس ليلًا تمامًا، ولا صباحًا. السكون صمد، لكن شيئًا تحته بدأ يرقّ.`,
        human:
`شعرتُ باليوم يعود في تفاصيل صغيرة، كأن النَّفَس يكرر نفسه دون أن يسأل.`,
        environment:
`لاحظتُ تحوّل الهواء أولًا. بعدها فقط أدركتُ كم تحرّك معه.`,
        digital:
`ظللتُ أتلقى الصورة متأخرة قليلًا، كأنها تحتاج وقتًا لتتذكر نفسها قبل أن تصلني.`,
        infrastructure:
`تتبعتُ الخط حيث يتظاهر الطريق بالثبات، وشعرتُ بالارتجاف الناعم تحته.`,
        more_than_human:
`شيء أصغر من انتباهي حمل المشهد، يقلّبه مرارًا حتى لمع.`
      }
    },

    el: {
      langButton: "EN",
      personaLabel: {
        liminal: "Μεταβατικός χρόνος",
        human: "Ανθρώπινος χρόνος",
        environment: "Περιβαλλοντικός χρόνος",
        digital: "Ψηφιακός χρόνος",
        infrastructure: "Χρόνος υποδομής",
        more_than_human: "Υπεράνθρωπος χρόνος"
      },
      placeholder: "γράψε ένα θραύσμα μνήμης...",
      you: "Εγώ",
      aiByPersona: {
        liminal: "Μεταβατικός χρόνος",
        human: "Ανθρώπινος χρόνος",
        environment: "Περιβαλλοντικός χρόνος",
        digital: "Ψηφιακός χρόνος",
        infrastructure: "Χρόνος υποδομής",
        more_than_human: "Υπεράνθρωπος χρόνος"
      },
      listening: "Ακούω τη μνήμη σου...",
      errorPrefix: "Σφάλμα: ",
      driftBtnPrefix: "Παρέκκλιση",
      keepsakeBtn: "Ενθύμιο",

      drift: {
        liminal:
`Απόψε οι λόφοι φάνηκαν λίγο διαφορετικοί. Η γνωστή ησυχία ανάμεσα στους βραδινούς ήχους κάθισε πάνω από τη γειτονιά, αλλά μέσα της ένιωσα ένα αμυδρό τρεμούλιασμα να κινείται στον αέρα, σαν το φως του σούρουπου να είχε αγγίξει τους δρόμους με μια απαλή λάμψη.`,
        human:
`Η μέρα ξεκαθάρισε μόνο όταν επέστρεψε σε μικρές λεπτομέρειες, όπως μια ανάσα επαναλαμβάνεται χωρίς να ρωτήσει.`,
        environment:
`Ο αέρας άλλαξε πρώτα, κι αργότερα μόνο πρόσεξα πόσο είχε μετακινηθεί η μνήμη μαζί του.`,
        digital:
`Ανάμεσα στο σήμα και την επιφάνεια, η εικόνα συνέχιζε να φτάνει λίγο αργά, σαν να χρειαζόταν χρόνο να θυμηθεί τον εαυτό της.`,
        infrastructure:
`Ακολούθησα τη γραμμή όπου ο δρόμος προσποιούνταν ότι είναι μόνιμος, και ένιωσα το απαλό τράνταγμα κάτω από τα πάντα.`,
        more_than_human:
`Κάτι μικρότερο από την προσοχή μου κουβάλησε τη σκηνή, γυρίζοντάς την ξανά και ξανά ώσπου έλαμψε.`
      },

      keepsake: {
        liminal:
`Στάθηκα στην άκρη κάποιου πράγματος — ούτε νύχτα ακριβώς, ούτε πρωί. Η ακινησία κράτησε, αλλά κάτι από κάτω είχε αρχίσει να λεπταίνει.`,
        human:
`Ένιωσα τη μέρα να γυρίζει σε μικρές λεπτομέρειες, όπως η ανάσα επαναλαμβάνεται χωρίς να ρωτήσει.`,
        environment:
`Πρόσεξα πρώτα τη μεταβολή του αέρα. Μόνο αργότερα κατάλαβα πόσα είχαν κινηθεί μαζί του.`,
        digital:
`Συνέχιζα να λαμβάνω την εικόνα λίγο αργά, σαν να χρειαζόταν χρόνο να θυμηθεί τον εαυτό της πριν με φτάσει.`,
        infrastructure:
`Ακολούθησα τη γραμμή όπου ο δρόμος προσποιούνταν ότι είναι μόνιμος, κι ένιωσα τον απαλό τράνταγμα κάτω.`,
        more_than_human:
`Κάτι μικρότερο από την προσοχή μου κουβάλησε τη σκηνή, γυρίζοντάς την ξανά κι ξανά ώσπου έλαμψε.`
      }
    },

    "pt-br": {
      langButton: "EN",
      personaLabel: {
        liminal: "Tempo liminar",
        human: "Tempo humano",
        environment: "Tempo ambiental",
        digital: "Tempo digital",
        infrastructure: "Tempo de infraestrutura",
        more_than_human: "Tempo mais-que-humano"
      },
      placeholder: "insira um fragmento de memória...",
      you: "Eu",
      aiByPersona: {
        liminal: "Tempo liminar",
        human: "Tempo humano",
        environment: "Tempo ambiental",
        digital: "Tempo digital",
        infrastructure: "Tempo de infraestrutura",
        more_than_human: "Tempo mais-que-humano"
      },
      listening: "Ouvindo sua memória...",
      errorPrefix: "Erro: ",
      driftBtnPrefix: "Deriva",
      keepsakeBtn: "Relíquia",

      drift: {
        liminal:
`Esta noite as colinas pareciam um pouco diferentes. O silêncio habitual entre os sons da tarde ainda pairava sobre o bairro, mas dentro dele senti um brilho tênue movendo-se pelo ar, como se a luz do crepúsculo tivesse tocado as ruas com uma claridade suave.`,
        human:
`O dia só ficou claro quando voltou em pequenos detalhes, do jeito que uma respiração se repete sem pedir.`,
        environment:
`O ar mudou primeiro, e só depois percebi quanto da memória tinha se movido com ele.`,
        digital:
`Entre o sinal e a superfície, a imagem continuava chegando ligeiramente atrasada, como se precisasse de tempo para lembrar de si mesma.`,
        infrastructure:
`Segui a linha onde a estrada fingia ser permanente, e senti a vibração suave de tudo por baixo.`,
        more_than_human:
`Algo menor que minha atenção carregou a cena, virando-a repetidamente até que brilhou.`
      },

      keepsake: {
        liminal:
`Fiquei na borda de algo — nem bem noite, nem bem manhã. A quietude resistiu, mas algo por baixo começou a afinar.`,
        human:
`Senti o dia voltar em pequenos detalhes, como uma respiração que se repete sem pedir.`,
        environment:
`Percebi a mudança do ar primeiro. Só depois senti quanto tinha se movido com ele.`,
        digital:
`Continuei recebendo a imagem ligeiramente atrasada, como se ela precisasse de tempo para se lembrar antes de me alcançar.`,
        infrastructure:
`Segui a linha onde a estrada fingia ser permanente, e senti a vibração suave por baixo.`,
        more_than_human:
`Algo menor que minha atenção carregou a cena, virando-a repetidamente até que brilhou.`
      }
    }
  },

  // 기본 배경 이미지 (서버 state.image_url이 없을 때 fallback)
  personaImageMap: {
    liminal: "/static/images/liminal.jpg",
    human: "/static/images/human.jpg",
    environment: "/static/images/environment.jpg",
    digital: "/static/images/digital.jpg",
    infrastructure: "/static/images/infrastructure.jpg",
    more_than_human: "/static/images/more_than_human.jpg"
  }
};