from subprocess import Popen, PIPE

import numpy as np

configs = {
    "1": {
        "avg_value": 0.0,
        "avg_value_by_weight": 0.0,
        "max_value": 0.0,
        "max_value_by_weight": 0.0,
        "min_value": 0.0,
        "min_value_by_weight": 0.0,
        "weight": -1.0
    },
    "2": {
        "avg_value": 0.5219235705710206,
        "avg_value_by_weight": -0.024107949765249215,
        "max_value": 0.8863486233863618,
        "max_value_by_weight": -0.7958049329959671,
        "min_value": -0.184317634155295,
        "min_value_by_weight": 0.4819669867076899,
        "weight": 0.8876146563027185
    },
    "3": {
        "avg_value": 0.6548237051440922,
        "avg_value_by_weight": -0.3628243360861998,
        "max_value": -0.7743859526107681,
        "max_value_by_weight": 0.8597961026841738,
        "min_value": -0.6521574300079507,
        "min_value_by_weight": 0.9068729090754273,
        "weight": -0.9913127647226463
    },
    "4": {
        "avg_value": -0.853128807575861,
        "avg_value_by_weight": 0.7975446428574846,
        "max_value": 0.8546166177913308,
        "max_value_by_weight": 0.0323944351266936,
        "min_value": -0.7301573137429609,
        "min_value_by_weight": -0.5405899301333965,
        "weight": -0.0520143363973955
    },
    "5": {
        "avg_value": 0.23047331243563174,
        "avg_value_by_weight": -0.06847617370511938,
        "max_value": -0.8529300229625347,
        "max_value_by_weight": -0.39018388947459015,
        "min_value": -0.032752589817819366,
        "min_value_by_weight": -0.22403745510235118,
        "weight": 0.7606510083384141
    },
    "6": {
        "avg_value": -0.7897658450899867,
        "avg_value_by_weight": -0.57474964829283,
        "max_value": -0.18252392521595207,
        "max_value_by_weight": -0.1674950199173375,
        "min_value": 0.9586317230502486,
        "min_value_by_weight": 0.9441481915965777,
        "weight": -0.9139393271673173
    },
    "7": {
        "avg_value": 0.7145929410146785,
        "avg_value_by_weight": -0.45953557535041845,
        "max_value": -0.02944476587722411,
        "max_value_by_weight": -0.45356411523834117,
        "min_value": -0.8454182444711582,
        "min_value_by_weight": 0.818763938771174,
        "weight": -0.4584560281700538
    },
    "8": {
        "avg_value": 0.6070459739939762,
        "avg_value_by_weight": -0.730972704809784,
        "max_value": 0.2467741195987594,
        "max_value_by_weight": -0.42155035614058844,
        "min_value": 0.6305112460378823,
        "min_value_by_weight": 0.6477285476058037,
        "weight": -0.919445295063682
    },
    "9": {
        "avg_value": -0.1828725070560835,
        "avg_value_by_weight": -0.1125276124704866,
        "max_value": 0.02286714886675245,
        "max_value_by_weight": 0.439953046274963,
        "min_value": -0.7947172747476177,
        "min_value_by_weight": 0.4998457218038175,
        "weight": -0.6124076266055032
    },
    "10": {
        "avg_value": 0.7199177012559737,
        "avg_value_by_weight": 0.18838602704753593,
        "max_value": -0.3402304250303667,
        "max_value_by_weight": -0.6070163311032055,
        "min_value": 0.5274035065832023,
        "min_value_by_weight": 0.611129131992286,
        "weight": 0.898959428020843
    },
    "11": {
        "avg_value": 0.3841007731881183,
        "avg_value_by_weight": 0.258285389874239,
        "max_value": 0.4715533976941817,
        "max_value_by_weight": -0.05144182042760859,
        "min_value": 0.03417643417028371,
        "min_value_by_weight": 0.8226335149504547,
        "weight": -0.9627283124481032
    },
    "12": {
        "avg_value": -0.8809054631169908,
        "avg_value_by_weight": -0.6612788374083114,
        "max_value": -0.04654270266924865,
        "max_value_by_weight": 0.958436761904466,
        "min_value": -0.701872468963495,
        "min_value_by_weight": 0.6207679421816148,
        "weight": -0.6412424226098734
    },
    "13": {
        "avg_value": 0.9509657772002353,
        "avg_value_by_weight": 0.8045947674978169,
        "max_value": -0.9545342401506987,
        "max_value_by_weight": -0.9196543555670984,
        "min_value": -0.37025335496012346,
        "min_value_by_weight": 0.5875553829018534,
        "weight": 0.9465675577967056
    },
    "14": {
        "avg_value": 0.5071136672410104,
        "avg_value_by_weight": 0.432075098438667,
        "max_value": -0.24031802084494647,
        "max_value_by_weight": -0.8320205626378727,
        "min_value": -0.5713603877167772,
        "min_value_by_weight": -0.6151014384137212,
        "weight": 0.7226736528174094
    },
    "15": {
        "avg_value": 0.8583941513695805,
        "avg_value_by_weight": 0.16753678005775718,
        "max_value": -0.1507652150748513,
        "max_value_by_weight": -0.8548780828416831,
        "min_value": 0.9013485624474069,
        "min_value_by_weight": 0.8457024414419958,
        "weight": -0.928583283143869
    },
    "16": {
        "avg_value": -0.7923441701444298,
        "avg_value_by_weight": 0.6323884725006435,
        "max_value": 0.8635859990973449,
        "max_value_by_weight": 0.0018452617679050043,
        "min_value": 0.8159495773736567,
        "min_value_by_weight": -0.7584112491303214,
        "weight": 0.5256475123511795
    },
    "17": {
        "avg_value": 0.5849935089575558,
        "avg_value_by_weight": -0.39552729816357246,
        "max_value": -0.926035889146976,
        "max_value_by_weight": -0.14053580129244825,
        "min_value": 0.5361660720094235,
        "min_value_by_weight": -0.028335787050499883,
        "weight": -0.05196938365238468
    },
    "18": {
        "avg_value": -0.7042138823391162,
        "avg_value_by_weight": 0.4657516720111854,
        "max_value": 0.7650321394018875,
        "max_value_by_weight": 0.3241825907211693,
        "min_value": 0.01186322863125655,
        "min_value_by_weight": -0.9248906965127213,
        "weight": -0.968855244573199
    },
    "19": {
        "avg_value": 0.5113558084728946,
        "avg_value_by_weight": -0.3393144371051684,
        "max_value": -0.8155985724799812,
        "max_value_by_weight": -0.5250054043050465,
        "min_value": 0.46688083225383825,
        "min_value_by_weight": 0.10842656330235734,
        "weight": 0.9511008179991123
    },
    "20": {
        "avg_value": -0.7033056080717273,
        "avg_value_by_weight": -0.3696637077951437,
        "max_value": -0.40832874928971963,
        "max_value_by_weight": -7.294516112743565e-05,
        "min_value": 0.3684950397035105,
        "min_value_by_weight": 0.09173509351209463,
        "weight": -0.9612006167779237
    },
    "21": {
        "avg_value": -0.5256279706694158,
        "avg_value_by_weight": -0.061973348194624966,
        "max_value": -0.7209160338622782,
        "max_value_by_weight": 0.8772804094943099,
        "min_value": -0.44168397223590383,
        "min_value_by_weight": 0.11570220555267086,
        "weight": 0.9465717368150963
    },
    "22": {
        "avg_value": -0.4720509731429887,
        "avg_value_by_weight": 0.7453979010062715,
        "max_value": -0.7976464437136175,
        "max_value_by_weight": 0.6385846741525785,
        "min_value": -0.3353730784530884,
        "min_value_by_weight": -0.9742043407803636,
        "weight": 0.9775154966454971
    },
    "23": {
        "avg_value": -0.7644995783417317,
        "avg_value_by_weight": 0.11020116917552669,
        "max_value": -0.28399700136051176,
        "max_value_by_weight": 0.14906881957330476,
        "min_value": 0.7881861888408415,
        "min_value_by_weight": 0.13172394087209982,
        "weight": 0.8780165474763375
    },
    "24": {
        "avg_value": 0.40932084265913815,
        "avg_value_by_weight": 0.987379362648316,
        "max_value": -0.5363820862795758,
        "max_value_by_weight": 0.17484458893044064,
        "min_value": 0.5673743450807371,
        "min_value_by_weight": 0.9489605123266902,
        "weight": -0.8944471889725674
    },
    "25": {
        "avg_value": -0.24834669833232792,
        "avg_value_by_weight": 0.22790601950242606,
        "max_value": -0.3160641730840994,
        "max_value_by_weight": 0.9565056030452073,
        "min_value": -0.7443666777735982,
        "min_value_by_weight": 0.6786127066837064,
        "weight": 0.6006893184219249
    },
    "26": {
        "avg_value": 0.1068487646934011,
        "avg_value_by_weight": -0.5013145487687514,
        "max_value": -0.4288994963852628,
        "max_value_by_weight": -0.3543766583812067,
        "min_value": -0.3063207654555298,
        "min_value_by_weight": 0.9113185374847892,
        "weight": 0.7300640219362893
    },
    "27": {
        "avg_value": -0.7800128635644306,
        "avg_value_by_weight": 0.946984208224019,
        "max_value": -0.5733356239075031,
        "max_value_by_weight": -0.5465436148013589,
        "min_value": -0.9445643604767082,
        "min_value_by_weight": 0.9878299055037547,
        "weight": 0.60252290936084
    },
    "28": {
        "avg_value": -0.7713992461238532,
        "avg_value_by_weight": -0.6554767473523888,
        "max_value": 0.6611489739843419,
        "max_value_by_weight": -0.04818798543354064,
        "min_value": -0.26486317359276335,
        "min_value_by_weight": 0.7474248367368224,
        "weight": 0.8417321001646287
    },
    "29": {
        "avg_value": -0.5854572506924705,
        "avg_value_by_weight": -0.7533718618153757,
        "max_value": 0.9402447140300891,
        "max_value_by_weight": 0.9231798471359676,
        "min_value": 0.2160261229598306,
        "min_value_by_weight": 0.42406056965534855,
        "weight": 0.7533010058956751
    },
    "30": {
        "avg_value": -0.5410234118154997,
        "avg_value_by_weight": 0.8017931931485436,
        "max_value": -0.15832939367928955,
        "max_value_by_weight": 0.17761913195130852,
        "min_value": -0.16654515009187176,
        "min_value_by_weight": -0.030942760523904544,
        "weight": 0.04749522059589584
    },
    "31": {
        "avg_value": -0.3449190246966356,
        "avg_value_by_weight": 0.7369626791787367,
        "max_value": -0.4055360792571179,
        "max_value_by_weight": 0.19379021404849173,
        "min_value": -0.07261291135296566,
        "min_value_by_weight": -0.4849862593751686,
        "weight": -0.9524050052233661
    },
    "32": {
        "avg_value": 0.8755227894977511,
        "avg_value_by_weight": -0.03584500341887975,
        "max_value": 0.6204120680921834,
        "max_value_by_weight": 0.2292316130310168,
        "min_value": 0.5870795212489544,
        "min_value_by_weight": -0.9857222561378634,
        "weight": 0.5168162795166049
    },
    "33": {
        "avg_value": 0.7007399414184163,
        "avg_value_by_weight": 0.7445091851551016,
        "max_value": -0.25517227362024464,
        "max_value_by_weight": 0.06962129949395002,
        "min_value": 0.8346703274765948,
        "min_value_by_weight": 0.9964117519162818,
        "weight": -0.29567265439865886
    },
    "34": {
        "avg_value": 0.18325468225593733,
        "avg_value_by_weight": -0.588443870776062,
        "max_value": 0.11904788799653643,
        "max_value_by_weight": 0.577121502123259,
        "min_value": -0.5159087342240307,
        "min_value_by_weight": -0.9499458575536006,
        "weight": 0.8181344763161547
    },
    "35": {
        "avg_value": 0.4334394667453749,
        "avg_value_by_weight": 0.6337713680491046,
        "max_value": -0.2633392679322829,
        "max_value_by_weight": 0.03859972288024105,
        "min_value": -0.3576213708914373,
        "min_value_by_weight": 0.9507922847989934,
        "weight": -0.08754093733902502
    },
    "36": {
        "avg_value": 0.9964977909274448,
        "avg_value_by_weight": 0.11690765996158015,
        "max_value": -0.370941167381986,
        "max_value_by_weight": 0.2650105892958132,
        "min_value": 0.8578686161975848,
        "min_value_by_weight": 0.44838837942652265,
        "weight": -0.37490204892832923
    },
    "37": {
        "avg_value": 0.7132854247649609,
        "avg_value_by_weight": -0.026219890007873836,
        "max_value": 0.381793961591383,
        "max_value_by_weight": -0.5821700068607387,
        "min_value": 0.952963534534522,
        "min_value_by_weight": 0.8661395624399002,
        "weight": -0.19552339993045975
    },
    "38": {
        "avg_value": 0.5605754401061416,
        "avg_value_by_weight": 0.9323400566311533,
        "max_value": -0.2651170309191846,
        "max_value_by_weight": -0.811368158091399,
        "min_value": -0.4994186046408442,
        "min_value_by_weight": 0.05016126840899915,
        "weight": 0.9130900925381311
    },
    "39": {
        "avg_value": 0.4614463847952346,
        "avg_value_by_weight": 0.5145212545033977,
        "max_value": 0.9522808039136559,
        "max_value_by_weight": -0.5741701675182103,
        "min_value": -0.8017064080354812,
        "min_value_by_weight": 0.5744524880498723,
        "weight": 0.6912253318179664
    },
    "40": {
        "avg_value": 0.6137152339637764,
        "avg_value_by_weight": 0.7859647736476314,
        "max_value": -0.4964060865345301,
        "max_value_by_weight": -0.5263106933643242,
        "min_value": -0.6298951538559863,
        "min_value_by_weight": -0.3979807342974062,
        "weight": 0.8307707257334893
    },
    "41": {
        "avg_value": -0.3054758301325706,
        "avg_value_by_weight": 0.19378792107152965,
        "max_value": -0.20353956705252707,
        "max_value_by_weight": -0.29510669635704667,
        "min_value": 0.7984003616609339,
        "min_value_by_weight": -0.035547072834866356,
        "weight": 0.9196885591921067
    },
    "42": {
        "avg_value": 0.978247935797762,
        "avg_value_by_weight": -0.23521816148629648,
        "max_value": -0.7607871305911111,
        "max_value_by_weight": 0.30860955501846954,
        "min_value": -0.20957153482320523,
        "min_value_by_weight": -0.7751579688477022,
        "weight": -0.8205484703929693
    },
    "43": {
        "avg_value": -0.2706330277366604,
        "avg_value_by_weight": 0.9508388065880153,
        "max_value": -0.4614594584826748,
        "max_value_by_weight": 0.11683090995910517,
        "min_value": -0.5354099935676473,
        "min_value_by_weight": 0.7510639041603244,
        "weight": 0.3831739166272172
    },
    "44": {
        "avg_value": 0.6297178004547703,
        "avg_value_by_weight": -0.008576292485014125,
        "max_value": 0.7117055726016774,
        "max_value_by_weight": 0.8977935811977042,
        "min_value": -0.8474072214646475,
        "min_value_by_weight": -0.6338244199230496,
        "weight": -0.2357932420843456
    },
    "45": {
        "avg_value": 0.5605754401061416,
        "avg_value_by_weight": 0.9323400566311533,
        "max_value": -0.2651170309191846,
        "max_value_by_weight": -0.9724845861977456,
        "min_value": -0.5498243901377694,
        "min_value_by_weight": 0.1603030278839246,
        "weight": 0.9130900925381311
    },
    "46": {
        "avg_value": 0.165518228878476,
        "avg_value_by_weight": 0.7474155840605508,
        "max_value": 0.8175064742174611,
        "max_value_by_weight": 0.6043716307364475,
        "min_value": 0.4327349430116936,
        "min_value_by_weight": -0.9418341963552523,
        "weight": -0.15388642648778417
    },
    "47": {
        "avg_value": 0.36293117386710994,
        "avg_value_by_weight": -0.6914100373814926,
        "max_value": -0.05491611362146531,
        "max_value_by_weight": -0.9938834112575965,
        "min_value": 0.40991401250322523,
        "min_value_by_weight": -0.037916013583855435,
        "weight": 0.9115595782393022
    },
    "48": {
        "avg_value": -0.24174591735225004,
        "avg_value_by_weight": -0.40921926916922513,
        "max_value": -0.7571603440572763,
        "max_value_by_weight": -0.09818827103851979,
        "min_value": 0.9791634268566656,
        "min_value_by_weight": 0.8428652569843997,
        "weight": -0.8917077434351057
    },
    "49": {
        "avg_value": -0.007815457529293024,
        "avg_value_by_weight": 0.7528066182430222,
        "max_value": -0.41769457300978696,
        "max_value_by_weight": 0.5402511539620394,
        "min_value": 0.7382320494962358,
        "min_value_by_weight": 0.5601025659183161,
        "weight": 0.6677038014665237
    },
    "50": {
        "avg_value": 0.3016036413339469,
        "avg_value_by_weight": -0.646710152108231,
        "max_value": 0.11744464363958862,
        "max_value_by_weight": -0.8727039938983676,
        "min_value": 0.40310676356160124,
        "min_value_by_weight": -0.037916013583855435,
        "weight": 0.9287734616701087
    },
    "51": {
        "avg_value": 0.03127741934090622,
        "avg_value_by_weight": 0.8517297360826204,
        "max_value": -0.5176445799159202,
        "max_value_by_weight": 0.14554241076473495,
        "min_value": 0.8047518859072842,
        "min_value_by_weight": -0.23719341114187664,
        "weight": 0.7722866073519885
    },
    "52": {
        "avg_value": -0.4948298578398169,
        "avg_value_by_weight": -0.27690934352340135,
        "max_value": 0.8107770430062367,
        "max_value_by_weight": -0.3569906965423695,
        "min_value": 0.08521344174312695,
        "min_value_by_weight": -0.08680839601195789,
        "weight": -0.888609242853043
    },
    "53": {
        "avg_value": 0.6571074262516226,
        "avg_value_by_weight": 0.5137221348425514,
        "max_value": -0.6590746119219377,
        "max_value_by_weight": 0.05691847027721697,
        "min_value": -0.886209636729981,
        "min_value_by_weight": 0.5120811109377283,
        "weight": 0.6548158806683808
    },
    "54": {
        "avg_value": -0.2045146739916197,
        "avg_value_by_weight": 0.883423550828293,
        "max_value": -0.3337893613832241,
        "max_value_by_weight": 0.15592171895985585,
        "min_value": -0.8700985534740912,
        "min_value_by_weight": -0.37981541662784724,
        "weight": 0.7782339084973386
    },
    "55": {
        "avg_value": -0.3267820377029724,
        "avg_value_by_weight": -0.20608775602579477,
        "max_value": -0.4347333166993843,
        "max_value_by_weight": 0.15906973946375347,
        "min_value": -0.9401007257849292,
        "min_value_by_weight": 0.6565378071371428,
        "weight": -0.6166727939110102
    },
    "56": {
        "avg_value": 0.6318896557751088,
        "avg_value_by_weight": 0.6083826139428055,
        "max_value": -0.6783235921661508,
        "max_value_by_weight": -0.9052752095016048,
        "min_value": -0.9326211483774958,
        "min_value_by_weight": -0.1966607704178729,
        "weight": 0.30729748949988567
    },
    "57": {
        "avg_value": 0.4488525703157318,
        "avg_value_by_weight": -0.6859681825448514,
        "max_value": -0.05943196685576757,
        "max_value_by_weight": -0.9088574854046662,
        "min_value": 0.4320236290540338,
        "min_value_by_weight": -0.130382205046904,
        "weight": 0.7841788101474816
    },
    "58": {
        "avg_value": -0.8107972674048565,
        "avg_value_by_weight": -0.516548339914563,
        "max_value": 0.5827082169205886,
        "max_value_by_weight": -0.8284969585678783,
        "min_value": 0.17863924919203567,
        "min_value_by_weight": -0.5453946961263059,
        "weight": -0.6572055097806384
    },
    "59": {
        "avg_value": -0.1209753558322324,
        "avg_value_by_weight": 0.9098001914897969,
        "max_value": -0.5538849734756282,
        "max_value_by_weight": 0.027664834480510825,
        "min_value": 0.3183135862853086,
        "min_value_by_weight": 0.46593408590773233,
        "weight": -0.25348456356291615
    },
    "60": {
        "avg_value": -0.9553528368692996,
        "avg_value_by_weight": 0.7472586344975731,
        "max_value": 0.12556090288896193,
        "max_value_by_weight": -0.809087883867307,
        "min_value": 0.43890082483965287,
        "min_value_by_weight": 0.30515680119810473,
        "weight": -0.6198851461302619
    },
    "61": {
        "avg_value": 0.3990691334085479,
        "avg_value_by_weight": -0.646710152108231,
        "max_value": 0.2710041857459051,
        "max_value_by_weight": -0.9262505058797432,
        "min_value": 0.6305734576102122,
        "min_value_by_weight": -0.18031283158236344,
        "weight": 0.7882372539164579
    },
    "62": {
        "avg_value": 0.24364567518149038,
        "avg_value_by_weight": -0.5012383417796866,
        "max_value": -0.540936104933742,
        "max_value_by_weight": -0.22710513484230588,
        "min_value": 0.37208598682532323,
        "min_value_by_weight": 0.8564718329376064,
        "weight": -0.9703223759964725
    },
    "63": {
        "avg_value": 0.4483094018811433,
        "avg_value_by_weight": 0.7283814863958733,
        "max_value": -0.1196398745983096,
        "max_value_by_weight": -0.940888478792823,
        "min_value": 0.790399364907103,
        "min_value_by_weight": -0.02824300528454382,
        "weight": 0.9914627317459577
    },
    "64": {
        "avg_value": 0.40961324403835975,
        "avg_value_by_weight": 0.8823938353955609,
        "max_value": -0.04347370085024127,
        "max_value_by_weight": -0.9660624687030995,
        "min_value": 0.8716494107805524,
        "min_value_by_weight": -0.01161009203213259,
        "weight": 0.934526139351401
    },
    "65": {
        "avg_value": -0.17192998766049983,
        "avg_value_by_weight": 0.09621842155822158,
        "max_value": 0.46318103939570987,
        "max_value_by_weight": -0.562913722439867,
        "min_value": 0.706099637542644,
        "min_value_by_weight": 0.9135280072204481,
        "weight": -0.2800834820892115
    },
    "66": {
        "avg_value": -0.9713745151928554,
        "avg_value_by_weight": 0.44480675128332314,
        "max_value": 0.2519877760565956,
        "max_value_by_weight": 0.13594553022551215,
        "min_value": -0.9699998661843785,
        "min_value_by_weight": 0.2715872276997515,
        "weight": 0.44256882316561375
    },
    "67": {
        "avg_value": 0.5794543376914514,
        "avg_value_by_weight": 0.96503581441832,
        "max_value": 0.20419081040314602,
        "max_value_by_weight": -0.1289158698593622,
        "min_value": -0.5519665955632327,
        "min_value_by_weight": -0.13542940701384043,
        "weight": 0.8689830809494832
    },
    "68": {
        "avg_value": 0.06811331807179877,
        "avg_value_by_weight": 0.6478348779818373,
        "max_value": 0.08629908739765946,
        "max_value_by_weight": 0.17916630058116034,
        "min_value": 0.793866973993016,
        "min_value_by_weight": 0.14220484376047993,
        "weight": 0.9820323201411039
    },
    "69": {
        "avg_value": 0.34463333279938246,
        "avg_value_by_weight": -0.3952964906907406,
        "max_value": -0.8539184596199065,
        "max_value_by_weight": 0.5040830755516024,
        "min_value": -0.02382577108636008,
        "min_value_by_weight": -0.2623841186290874,
        "weight": 0.10052982630304186
    },
    "70": {
        "avg_value": 0.18602485023184911,
        "avg_value_by_weight": 0.6193986478554558,
        "max_value": -0.16912893757332292,
        "max_value_by_weight": 0.2438408131349199,
        "min_value": -0.49220302093267754,
        "min_value_by_weight": 0.9457560318043374,
        "weight": 0.9519059521193896
    },
    "71": {
        "avg_value": 0.09006751990554163,
        "avg_value_by_weight": 0.30573416275365717,
        "max_value": -0.27434035565843906,
        "max_value_by_weight": 0.22592472861247792,
        "min_value": 0.5059572246121748,
        "min_value_by_weight": 0.017250768494844637,
        "weight": 0.8833031079771507
    },
    "72": {
        "avg_value": 0.559156564606009,
        "avg_value_by_weight": 0.8756090413924018,
        "max_value": 0.838792321811427,
        "max_value_by_weight": -0.5654177882041198,
        "min_value": 0.45186397290068614,
        "min_value_by_weight": 0.009189668093722458,
        "weight": 0.794891400568545
    },
    "73": {
        "avg_value": 0.8556396925642273,
        "avg_value_by_weight": 0.8412216385462541,
        "max_value": 0.27472223949870944,
        "max_value_by_weight": -0.554148374166589,
        "min_value": -0.583291936481874,
        "min_value_by_weight": 0.7624345539809281,
        "weight": 0.8192408786285399
    },
    "74": {
        "avg_value": 0.02185274192361719,
        "avg_value_by_weight": 0.6565536672526566,
        "max_value": 0.0720602603821856,
        "max_value_by_weight": -0.11571141164288479,
        "min_value": -0.3028250253557945,
        "min_value_by_weight": 0.6785419764642138,
        "weight": -0.9478368614667081
    },
    "75": {
        "avg_value": -0.9112151478390897,
        "avg_value_by_weight": 0.09841239453935935,
        "max_value": -0.4999751410221629,
        "max_value_by_weight": -0.4863149539454943,
        "min_value": -0.5949376084081313,
        "min_value_by_weight": -0.8935598775895799,
        "weight": -0.5733935492946953
    },
    "76": {
        "avg_value": 0.6954940263257836,
        "avg_value_by_weight": -0.07598981347465084,
        "max_value": -0.2302646205171992,
        "max_value_by_weight": 0.09378372828536219,
        "min_value": -0.7539070323473758,
        "min_value_by_weight": 0.4844349238449819,
        "weight": -0.20350196318254454
    },
    "77": {
        "avg_value": 0.795663479812748,
        "avg_value_by_weight": 0.9391062357689182,
        "max_value": 0.254517067540337,
        "max_value_by_weight": 0.2320940242285514,
        "min_value": -0.6347917447990643,
        "min_value_by_weight": -0.13542940701384043,
        "weight": 0.8285767667545605
    },
    "78": {
        "avg_value": -0.6213545088061065,
        "avg_value_by_weight": 0.35030105251307964,
        "max_value": 0.7315494502362803,
        "max_value_by_weight": 0.15295195816662344,
        "min_value": -0.14997761835377,
        "min_value_by_weight": -0.018977821718046295,
        "weight": 0.18167941580635305
    },
    "79": {
        "avg_value": 0.6150590862374168,
        "avg_value_by_weight": 0.8815351728310943,
        "max_value": 0.35646548614361695,
        "max_value_by_weight": 0.5013530806252309,
        "min_value": -0.14624850583926507,
        "min_value_by_weight": -0.026158135337704724,
        "weight": 0.8425904968644955
    },
    "80": {
        "avg_value": -0.4145256005359632,
        "avg_value_by_weight": -0.3571377134669993,
        "max_value": -0.18020085240956174,
        "max_value_by_weight": -0.26346333558513235,
        "min_value": -0.6985528160567158,
        "min_value_by_weight": -0.9482486304737785,
        "weight": -0.09012998729664567
    },
    "81": {
        "avg_value": 0.9424567410025186,
        "avg_value_by_weight": 0.8761962665437015,
        "max_value": 0.4879493131344257,
        "max_value_by_weight": -0.9890695290304852,
        "min_value": -0.5872367783340176,
        "min_value_by_weight": 0.7624345539809281,
        "weight": 0.8192408786285399
    },
    "82": {
        "avg_value": -0.564391498600139,
        "avg_value_by_weight": 0.002956978607520089,
        "max_value": 0.3621428388544785,
        "max_value_by_weight": -0.3781013413269405,
        "min_value": 0.2629210969453304,
        "min_value_by_weight": 0.5067577483361825,
        "weight": 0.9727832621628036
    },
    "83": {
        "avg_value": -0.5597523988786723,
        "avg_value_by_weight": 0.4635171557242601,
        "max_value": -0.23732398015788148,
        "max_value_by_weight": 0.5524770501241556,
        "min_value": 0.8662509551520969,
        "min_value_by_weight": 0.3244300924068315,
        "weight": -0.989699899895198
    },
    "84": {
        "avg_value": 0.6881886222366318,
        "avg_value_by_weight": 0.8408690903012546,
        "max_value": -0.36250804491876165,
        "max_value_by_weight": -0.8268926348593717,
        "min_value": -0.31482832386959814,
        "min_value_by_weight": -0.0832988561723188,
        "weight": 0.7283738254764942
    },
    "85": {
        "avg_value": 0.4396726462921292,
        "avg_value_by_weight": 0.7385499571259957,
        "max_value": -0.39265706762987984,
        "max_value_by_weight": -0.9019537718445154,
        "min_value": -0.5954216038452222,
        "min_value_by_weight": 0.029306602704500806,
        "weight": 0.7476854706377944
    },
    "86": {
        "avg_value": -0.38548697325243353,
        "avg_value_by_weight": 0.14081852392285943,
        "max_value": 0.13733459948474014,
        "max_value_by_weight": 0.536653021857904,
        "min_value": -0.29628981535018784,
        "min_value_by_weight": 0.25699975841367295,
        "weight": -0.4781312853725792
    },
    "87": {
        "avg_value": -0.8157192525574972,
        "avg_value_by_weight": -0.38617076180093957,
        "max_value": -0.10995429939623813,
        "max_value_by_weight": -0.10604111805069216,
        "min_value": -0.41482423213390995,
        "min_value_by_weight": 0.6002491466012103,
        "weight": 0.433012550259517
    },
    "88": {
        "avg_value": -0.6065568498537095,
        "avg_value_by_weight": 0.8553259591655828,
        "max_value": -0.5774839552625288,
        "max_value_by_weight": 0.9896437745389333,
        "min_value": -0.5048733383939956,
        "min_value_by_weight": -0.13891249831515196,
        "weight": 0.04143354736741389
    },
    "89": {
        "avg_value": 0.8839543448772296,
        "avg_value_by_weight": 0.8549379725656379,
        "max_value": 0.6302646642731169,
        "max_value_by_weight": 0.2949844761963769,
        "min_value": -0.6890296587191186,
        "min_value_by_weight": -0.11186289009309291,
        "weight": 0.7576245010341398
    },
    "90": {
        "avg_value": 0.23170801155285625,
        "avg_value_by_weight": -0.9314237840917332,
        "max_value": 0.048055816338069546,
        "max_value_by_weight": -0.35139445116237344,
        "min_value": 0.034448615030459706,
        "min_value_by_weight": 0.6884246108204661,
        "weight": -0.0209345758658126
    },
    "91": {
        "avg_value": 0.9547991324215004,
        "avg_value_by_weight": 0.8678515669076874,
        "max_value": 0.254517067540337,
        "max_value_by_weight": 0.34208866621747,
        "min_value": -0.5092601838326578,
        "min_value_by_weight": -0.10039640191238774,
        "weight": 0.766342355697955
    },
    "92": {
        "avg_value": 0.03972764194205669,
        "avg_value_by_weight": 0.20832676215563128,
        "max_value": -0.24776007283887713,
        "max_value_by_weight": 0.20767645490874576,
        "min_value": 0.1358256885017013,
        "min_value_by_weight": 0.48198658782626635,
        "weight": 0.7979238191090994
    },
    "93": {
        "avg_value": 0.8839543448772296,
        "avg_value_by_weight": 0.8004888966983179,
        "max_value": 0.26437427839004446,
        "max_value_by_weight": 0.14390674758492494,
        "min_value": -0.8846344878792949,
        "min_value_by_weight": -0.12781809685956957,
        "weight": 0.6484087524171624
    },
    "94": {
        "avg_value": 0.9834219159124671,
        "avg_value_by_weight": 0.8050441404247033,
        "max_value": 0.21495750430341465,
        "max_value_by_weight": 0.39201943495760583,
        "min_value": -0.3711578411520785,
        "min_value_by_weight": -0.18136678057796196,
        "weight": 0.6630001013168061
    },
    "95": {
        "avg_value": 0.7615347879748582,
        "avg_value_by_weight": 0.5241830955196154,
        "max_value": 0.0043340749737470485,
        "max_value_by_weight": 0.3995232005647569,
        "min_value": 0.09831035257773402,
        "min_value_by_weight": 0.8369222143884283,
        "weight": -0.5855134804413811
    },
    "96": {
        "avg_value": 0.926722309473548,
        "avg_value_by_weight": 0.9668788567291189,
        "max_value": 0.277626315852721,
        "max_value_by_weight": 0.2929373646221023,
        "min_value": -0.7226011187696041,
        "min_value_by_weight": -0.25234107710202036,
        "weight": 0.6669322008036978
    },
    "97": {
        "avg_value": 0.1919874528787111,
        "avg_value_by_weight": 0.05787795095417958,
        "max_value": 0.004639664097866758,
        "max_value_by_weight": 0.7680040635064531,
        "min_value": 0.9055709072073297,
        "min_value_by_weight": -0.28384254032443534,
        "weight": -0.9730656675848104
    },
    "98": {
        "avg_value": 0.8363504512343594,
        "avg_value_by_weight": 0.9435682433157397,
        "max_value": 0.3399334597278283,
        "max_value_by_weight": 0.45413990882203925,
        "min_value": -0.523807550328146,
        "min_value_by_weight": -0.15431432881177898,
        "weight": 0.6236461003492955
    },
    "99": {
        "avg_value": 0.8471313961052562,
        "avg_value_by_weight": 0.9679354212268778,
        "max_value": 0.524766808038766,
        "max_value_by_weight": 0.4757476485569345,
        "min_value": -0.5461870764979833,
        "min_value_by_weight": -0.1946393489945164,
        "weight": 0.6322127720531194
    },
    "100": {
        "avg_value": 0.8473313457199052,
        "avg_value_by_weight": 0.8885786300980316,
        "max_value": 0.23391137203231782,
        "max_value_by_weight": 0.2949844761963769,
        "min_value": -0.5273751128262045,
        "min_value_by_weight": -0.04112734425325537,
        "weight": 0.6287238872589076
    },
    "101": {
        "avg_value": 0.8121681817821518,
        "avg_value_by_weight": 0.8513671622602168,
        "max_value": 0.20529497654222362,
        "max_value_by_weight": 0.032007266156822256,
        "min_value": -0.44810837590483177,
        "min_value_by_weight": -0.30649328361142425,
        "weight": 0.6264788994213042
    },
    "102": {
        "avg_value": 0.8441268908407438,
        "avg_value_by_weight": 0.8420189781599419,
        "max_value": 0.13171182363112743,
        "max_value_by_weight": -0.14477339091972563,
        "min_value": 0.914949662718479,
        "min_value_by_weight": 0.03188192743390528,
        "weight": 0.6223825858287184
    },
    "103": {
        "avg_value": -0.16155001996345897,
        "avg_value_by_weight": -0.4496652658080147,
        "max_value": -0.7116195283644766,
        "max_value_by_weight": -0.9528238017734059,
        "min_value": -0.6621983121737238,
        "min_value_by_weight": -0.1127446906550531,
        "weight": -0.6328903999437479
    },
    "104": {
        "avg_value": 0.5925942490099907,
        "avg_value_by_weight": 0.5674995238302685,
        "max_value": 0.22565237428679552,
        "max_value_by_weight": 0.4439326458782442,
        "min_value": 0.0939404098920098,
        "min_value_by_weight": -0.8182098678791732,
        "weight": 0.9628724748485566
    },
    "105": {
        "avg_value": -0.5090018212991354,
        "avg_value_by_weight": 0.10845560910638685,
        "max_value": 0.3135290976301677,
        "max_value_by_weight": 0.0019924033636364857,
        "min_value": 0.07109868727582391,
        "min_value_by_weight": -0.6072677783632563,
        "weight": 0.0901145349583039
    },
    "106": {
        "avg_value": 0.8994746036464745,
        "avg_value_by_weight": 0.8910063453876176,
        "max_value": 0.3202496056433002,
        "max_value_by_weight": 0.18443833420268918,
        "min_value": -0.4497588870677198,
        "min_value_by_weight": 0.17761328597635262,
        "weight": 0.6210603087931443
    },
    "107": {
        "avg_value": 0.8988869787520295,
        "avg_value_by_weight": 0.8633913087335632,
        "max_value": 0.26288064202688055,
        "max_value_by_weight": 0.3080047776963293,
        "min_value": -0.4240028570925858,
        "min_value_by_weight": 0.09270394145212468,
        "weight": 0.6215343559034587
    },
    "108": {
        "avg_value": 0.9709455168119132,
        "avg_value_by_weight": 0.8708778247263202,
        "max_value": 0.29734781459860216,
        "max_value_by_weight": 0.4757476485569345,
        "min_value": -0.4787677090793747,
        "min_value_by_weight": -0.08896463426441747,
        "weight": 0.6172363516804675
    },
    "109": {
        "avg_value": 0.9797967398499539,
        "avg_value_by_weight": 0.9641002750357714,
        "max_value": -0.018538940430671058,
        "max_value_by_weight": 0.4857040534700656,
        "min_value": -0.48516189731139425,
        "min_value_by_weight": 0.057807833865547886,
        "weight": 0.6079009156153936
    },
    "110": {
        "avg_value": -0.4973823604306544,
        "avg_value_by_weight": 0.6370040697894563,
        "max_value": -0.7562155938099366,
        "max_value_by_weight": 0.40889710645085,
        "min_value": -0.47130582566614443,
        "min_value_by_weight": -0.19798315222369123,
        "weight": 0.3739547617495085
    },
    "111": {
        "avg_value": 0.9941441722418471,
        "avg_value_by_weight": 0.8863014308639678,
        "max_value": -0.0005787968814192634,
        "max_value_by_weight": 0.23442359392813072,
        "min_value": -0.14624850583926507,
        "min_value_by_weight": -0.026158135337704724,
        "weight": 0.6140032747434727
    },
    "112": {
        "avg_value": 0.8473313457199052,
        "avg_value_by_weight": 0.892097694849874,
        "max_value": 0.26379620102879797,
        "max_value_by_weight": 0.43097395128777105,
        "min_value": -0.24537875726629044,
        "min_value_by_weight": -0.008667839780895692,
        "weight": 0.6100133223458404
    },
    "113": {
        "avg_value": 0.666083877461465,
        "avg_value_by_weight": 0.6205134851570597,
        "max_value": -0.40759864645434774,
        "max_value_by_weight": -0.4153953091120586,
        "min_value": 0.8063690103493208,
        "min_value_by_weight": 0.850814384914448,
        "weight": 0.40069932857244495
    },
    "114": {
        "avg_value": 0.8936905064447813,
        "avg_value_by_weight": 0.8758996789408089,
        "max_value": 0.20267115443806327,
        "max_value_by_weight": 0.44348891429396864,
        "min_value": -0.3080334052908015,
        "min_value_by_weight": -0.202858242153381,
        "weight": 0.6092656148130509
    },
    "115": {
        "avg_value": -0.04390344492648035,
        "avg_value_by_weight": -0.5392912247166413,
        "max_value": 0.9903905111742237,
        "max_value_by_weight": -0.7610091250238862,
        "min_value": -0.7207205374438486,
        "min_value_by_weight": -0.9714030531757993,
        "weight": 0.5014263210031094
    },
    "116": {
        "avg_value": 0.8838580541574244,
        "avg_value_by_weight": 0.9684338219244057,
        "max_value": 0.22042533045286805,
        "max_value_by_weight": 0.4708483278745972,
        "min_value": -0.2703379128773107,
        "min_value_by_weight": 0.7107038356448725,
        "weight": 0.6087409047836436
    },
    "117": {
        "avg_value": 0.9492022937446125,
        "avg_value_by_weight": 0.8651188874438369,
        "max_value": 0.25513544868000815,
        "max_value_by_weight": 0.4573442963685219,
        "min_value": -0.29339411121017034,
        "min_value_by_weight": 0.21868850737205792,
        "weight": 0.6100830243661024
    },
    "118": {
        "avg_value": -0.3787165327864159,
        "avg_value_by_weight": -0.9604270309189709,
        "max_value": -0.1168166852062884,
        "max_value_by_weight": 0.5448873743626674,
        "min_value": -0.2213976614628277,
        "min_value_by_weight": -0.24011051361316826,
        "weight": -0.7139248903042374
    },
    "119": {
        "avg_value": 0.890900000412602,
        "avg_value_by_weight": 0.9750478361346018,
        "max_value": 0.21803937154199438,
        "max_value_by_weight": 0.4431804509369035,
        "min_value": -0.5226417263453356,
        "min_value_by_weight": -0.04859485486353721,
        "weight": 0.6074233616719942
    },
    "120": {
        "avg_value": 0.7473023736740818,
        "avg_value_by_weight": 0.994963421990446,
        "max_value": 0.22105369820046872,
        "max_value_by_weight": 0.47221040570721673,
        "min_value": -0.4432247603224092,
        "min_value_by_weight": -0.04207225516991786,
        "weight": 0.607558892885355
    },
    "121": {
        "avg_value": 0.7315427265264043,
        "avg_value_by_weight": 0.9228504354217357,
        "max_value": 0.2229860698127566,
        "max_value_by_weight": 0.3863273689607898,
        "min_value": -0.7467278468786674,
        "min_value_by_weight": -0.06861791799869699,
        "weight": 0.606560839443439
    },
    "122": {
        "avg_value": 0.9816444080655491,
        "avg_value_by_weight": 0.7991821049763801,
        "max_value": 0.2140287579854372,
        "max_value_by_weight": 0.32289445011283036,
        "min_value": -0.49479611487809505,
        "min_value_by_weight": -0.6803712747234522,
        "weight": 0.6043934114446088
    },
    "123": {
        "avg_value": -0.6486540442391728,
        "avg_value_by_weight": 0.36243596534829425,
        "max_value": -0.3172957656923814,
        "max_value_by_weight": -0.32306884944895264,
        "min_value": 0.7135539668520599,
        "min_value_by_weight": 0.5228859315272689,
        "weight": -0.10824784944668964
    },
    "124": {
        "avg_value": 0.5632891036155654,
        "avg_value_by_weight": -0.4990566270424883,
        "max_value": -0.4315840548524832,
        "max_value_by_weight": 0.8941764146824387,
        "min_value": 0.5678273545764769,
        "min_value_by_weight": -0.5098253452818315,
        "weight": 0.8644281838702397
    },
    "125": {
        "avg_value": -0.41432363592064325,
        "avg_value_by_weight": -0.0833431633188717,
        "max_value": -0.1735184874543707,
        "max_value_by_weight": 0.5906137430373404,
        "min_value": -0.15280539560502504,
        "min_value_by_weight": -0.467794452162384,
        "weight": 0.5432189924279482
    },
    "126": {
        "avg_value": -0.2492479246639301,
        "avg_value_by_weight": 0.05182582586857021,
        "max_value": -0.4287108638290691,
        "max_value_by_weight": -0.2685264394976601,
        "min_value": -0.21589149865834623,
        "min_value_by_weight": -0.7261425219672604,
        "weight": -0.40633121416711604
    },
    "127": {
        "avg_value": 0.322637836997276,
        "avg_value_by_weight": -0.5107699450276211,
        "max_value": -0.13219113722565767,
        "max_value_by_weight": 0.1632009635674776,
        "min_value": 0.7446048092229052,
        "min_value_by_weight": -0.06138465563521289,
        "weight": 0.053904380687016085
    },
    "128": {
        "avg_value": 0.03867298933902408,
        "avg_value_by_weight": 0.04437520054915378,
        "max_value": -0.26543872275396785,
        "max_value_by_weight": 0.07449599528907513,
        "min_value": -0.03389506062142589,
        "min_value_by_weight": -0.22283250200328286,
        "weight": 0.07164669619258124
    },
    "129": {
        "avg_value": -0.5757754296720574,
        "avg_value_by_weight": 0.8981775908994853,
        "max_value": 0.5840855632885957,
        "max_value_by_weight": 0.613245022037479,
        "min_value": -0.9503563622901097,
        "min_value_by_weight": -0.4989557191507681,
        "weight": 0.10355716028038287
    },
    "130": {
        "avg_value": 0.5889112621232462,
        "avg_value_by_weight": 0.8522734864304609,
        "max_value": 0.22662399298116953,
        "max_value_by_weight": 0.4482579522992409,
        "min_value": -0.42121205615279345,
        "min_value_by_weight": -0.016074141244272866,
        "weight": 0.6151907730353703
    },
    "131": {
        "avg_value": 0.7161440495352638,
        "avg_value_by_weight": 0.9530631116698178,
        "max_value": 0.2298830754238339,
        "max_value_by_weight": 0.4259870976163236,
        "min_value": -0.4287188681689639,
        "min_value_by_weight": -0.7126752967210814,
        "weight": 0.6158130831000177
    },
    "132": {
        "avg_value": 0.8839543448772296,
        "avg_value_by_weight": 0.8004888966983179,
        "max_value": 0.24541618991802383,
        "max_value_by_weight": 0.39228637165917446,
        "min_value": -0.796965530812917,
        "min_value_by_weight": 0.0903887229160929,
        "weight": 0.7913837689955543
    },
    "133": {
        "avg_value": 0.8558859366416041,
        "avg_value_by_weight": 0.8003944410338852,
        "max_value": 0.254517067540337,
        "max_value_by_weight": 0.2709258477594254,
        "min_value": -0.6061170812632976,
        "min_value_by_weight": 0.13407425115106575,
        "weight": 0.7913906439006946
    },
    "134": {
        "avg_value": 0.7999217154543581,
        "avg_value_by_weight": 0.8016393669265778,
        "max_value": 0.24125859006994443,
        "max_value_by_weight": 0.2788196006298873,
        "min_value": -0.6890296587191186,
        "min_value_by_weight": 0.04951537339141954,
        "weight": 0.7797345792883192
    },
    "135": {
        "avg_value": 0.5909431038114665,
        "avg_value_by_weight": -0.18082509738969899,
        "max_value": -0.48499428790812815,
        "max_value_by_weight": -0.20477315536271412,
        "min_value": 0.11151209189795064,
        "min_value_by_weight": -0.7023446792482013,
        "weight": 0.3467869441917708
    },
    "136": {
        "avg_value": -0.017742036258943106,
        "avg_value_by_weight": 0.2532796539398583,
        "max_value": -0.6146221679100545,
        "max_value_by_weight": 0.22079409535010308,
        "min_value": 0.24385753261824505,
        "min_value_by_weight": 0.6772801016260299,
        "weight": -0.12423397405967762
    },
    "137": {
        "avg_value": -0.9894349577731687,
        "avg_value_by_weight": 0.031964466392851554,
        "max_value": 0.09895859981588573,
        "max_value_by_weight": 0.19244543709499062,
        "min_value": -0.5009692841105999,
        "min_value_by_weight": -0.7450272363373114,
        "weight": -0.9853140790036596
    },
    "138": {
        "avg_value": 0.9613596391717751,
        "avg_value_by_weight": 0.9841277488532618,
        "max_value": 0.24832769145757339,
        "max_value_by_weight": 0.34342084588781896,
        "min_value": -0.740149360687296,
        "min_value_by_weight": 0.1028883736303734,
        "weight": 0.603664241118842
    },
    "139": {
        "avg_value": 0.915574888326673,
        "avg_value_by_weight": 0.9688520298948697,
        "max_value": 0.26288064202688055,
        "max_value_by_weight": 0.39851088330740114,
        "min_value": -0.4240028570925858,
        "min_value_by_weight": 0.04642766779941088,
        "weight": 0.603755192062789
    },
    "140": {
        "avg_value": 0.9407407630932749,
        "avg_value_by_weight": 0.9811354012751614,
        "max_value": 0.22970224270741113,
        "max_value_by_weight": 0.5045854707758359,
        "min_value": -0.718973755136397,
        "min_value_by_weight": -0.4771597710436305,
        "weight": 0.6037516774131286
    },
    "141": {
        "avg_value": 0.8839543448772296,
        "avg_value_by_weight": 0.8664797581453783,
        "max_value": 0.3764539563463931,
        "max_value_by_weight": 0.6449124604441758,
        "min_value": -0.6720204268187097,
        "min_value_by_weight": 0.09455984834889608,
        "weight": 0.975321179295852
    },
    "142": {
        "avg_value": 0.9121951178578529,
        "avg_value_by_weight": 0.9600899074190257,
        "max_value": 0.45800921358925106,
        "max_value_by_weight": 0.6284530742125609,
        "min_value": -0.5949328048830239,
        "min_value_by_weight": -0.04329727907117198,
        "weight": 0.9978349062810583
    },
    "143": {
        "avg_value": -0.01609063899091101,
        "avg_value_by_weight": 0.6505804706109715,
        "max_value": 0.7468356199826938,
        "max_value_by_weight": -0.6297590090475933,
        "min_value": 0.7452530790410588,
        "min_value_by_weight": 0.11268860923479229,
        "weight": -0.5169235737145375
    },
    "144": {
        "avg_value": 0.8993954361145506,
        "avg_value_by_weight": 0.8462727258129443,
        "max_value": 0.2868158787090638,
        "max_value_by_weight": 0.36405449175485605,
        "min_value": -0.7697833464472132,
        "min_value_by_weight": -0.04446897075816081,
        "weight": 0.7848517157433248
    },
    "145": {
        "avg_value": 0.7167891135520477,
        "avg_value_by_weight": -0.8183842979055473,
        "max_value": -0.2755990064273346,
        "max_value_by_weight": -0.7135715138274914,
        "min_value": -0.3797369641584527,
        "min_value_by_weight": -0.9958829702226086,
        "weight": -0.857385429688859
    }
}

# instances = [f"./3_20/train/kp_7_3_20_{i}.dat" for i in range(250)]
# print(instances[0])

# cmd = f"./multiobj {instances[0]} ./sample_weighted.csv 1 1 0 0 0 0 0 0 0 0 0"

# io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)
# (stdout_, stderr_) = io.communicate()

# # print(stdout_, stderr_)
# stdout_str = stdout_.decode('utf-8')

# if len(stdout_str):
#     status, result = stdout_str.split(":")
#     print(result)
#     if status == "Solved":
#         print(np.sum(list(map(float, stdout_str.strip().split(',')[-3:]))))


# instances = [f"../../kp/3_20/train/kp_7_3_20_0.dat" for i in range(250)]
instances = ["../../data/kp/3_20/train/kp_7_3_20_0.dat"]

# wt_dict = {"max_avg_value": 0.4500185954509764,
#            "max_avg_value_by_weight": 0.47900710822884596,
#            "max_max_value": 0.10105274422511748,
#            "max_max_value_by_weight": 0.14390745990629927,
#            "max_min_value": 0.2909215164563521,
#            "max_weight": 0.9182967088730671,
#            "min_avg_value": 0.42199285253811414,
#            "min_max_value": 0.525232096333076,
#            "min_min_value": 0.1087829211786774,
#            "min_weight": 0.9089081579717498}

wt_lst = ['weight',
          'avg_value',
          'max_value',
          'min_value',
          'avg_value_by_weight',
          'max_value_by_weight',
          'min_value_by_weight']

for k in configs.keys():

    # print(sol_list[0]["incumbent"])
    # wt_dict = sol["incumbent"]
    # wt_dict = {
    #     "max_avg_value": 0.11549915736265404,
    #     "max_avg_value_by_weight": 0.8621751314406639,
    #     "max_max_value": 0.0822246980956244,
    #     "max_max_value_by_weight": 0.8158566174992625,
    #     "max_min_value": 0.5339267511148671,
    #     "max_weight": 0.18449223611684668,
    #     "min_avg_value": 0.5411655660326893,
    #     "min_max_value": 0.8911638687639566,
    #     "min_min_value": 0.5239021574884885,
    #     "min_weight": 0.9382864770780639
    # }

    wt_str = " ".join([f'{configs[k][feat]}' for feat in wt_lst])
    # print(wt_str)

    total_time = []
    for instance in instances:
        cmd = f"./multiobj {instance}"
        cmd = cmd + " " + wt_str

        # print(cmd)
        io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)
        (stdout_, stderr_) = io.communicate()

        # print(stdout_, stderr_)
        stdout_str = stdout_.decode('utf-8')
        stderr_str = stderr_.decode('utf-8')

        print(stdout_str)
        print(stderr_str)

        print(np.sum(list(map(float, stdout_str.strip().split(',')[-3:]))))

        # if len(stdout_str):
        #     status, result = stdout_str.split(":")
        #     # print(instance, result)
        #     if status == "Solved":
        #         total_time.append(
        #             np.sum(list(map(float, stdout_str.strip().split(',')[-3:]))))

    # print(len(total_time), np.mean(total_time),
    #       np.std(total_time), np.median(total_time))
