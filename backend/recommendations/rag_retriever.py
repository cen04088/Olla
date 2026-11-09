"""RAG (Retrieval-Augmented Generation) 문서 검색 엔진.

사용자 질문과 관련된 산행 안전 지식, 탐방로 정보, 재난위험 데이터를 검색하여
LLM 컨텍스트로 제공합니다.
"""
import re
from array import array
from functools import lru_cache

# ── 정적 지식 베이스 ──────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    {
        "id": "kb-001",
        "category": "emergency",
        "title": "산에서 조난 시 대처",
        "content": (
            "산에서 조난 시 119에 신고하고 현재 위치를 최대한 정확히 알려주세요. "
            "GPS 좌표, 주변 지형지물(봉우리, 계곡, 대피소), 탐방로 이름을 말하세요. "
            "체온 유지를 위해 비상 은박 담요를 사용하고, 가능하면 바람을 피할 수 있는 곳에서 대기하세요. "
            "119 연결이 안 되면 산림청 산불·산사태 신고 전화 1688-9119를 이용하세요."
        ),
        "keywords": ["조난", "구조", "119", "신고", "대피", "응급", "긴급", "실종"],
    },
    {
        "id": "kb-002",
        "category": "emergency",
        "title": "산에서 골절·부상 응급처치",
        "content": (
            "골절이 의심되면 부목(나뭇가지, 등산스틱)으로 고정하고 움직임을 최소화하세요. "
            "출혈이 있으면 깨끗한 천으로 압박지혈하고 119에 즉시 신고하세요. "
            "혼자 이동하려 하지 말고 구조대를 기다리세요. "
            "산악 골절은 2차 부상 위험이 크므로 전문 구조대의 도움이 필요합니다."
        ),
        "keywords": ["골절", "부상", "응급처치", "부목", "출혈", "지혈", "다치다", "접질리다"],
    },
    {
        "id": "kb-003",
        "category": "emergency",
        "title": "탈진·저체온증 대처",
        "content": (
            "탈진 시 그늘진 곳에서 휴식하고 수분과 간식을 섭취하세요. "
            "저체온증은 체온이 35°C 이하로 떨어질 때 발생하며, 오한·의식 저하·말 더듬음이 증상입니다. "
            "젖은 옷을 벗기고 여분 옷이나 비상 은박 담요로 감싸 체온을 올리세요. "
            "따뜻한 음료(알코올 제외)를 마시게 하고 즉시 하산하거나 119에 신고하세요."
        ),
        "keywords": ["탈진", "저체온", "체온", "오한", "응급", "탈수", "피로"],
    },
    {
        "id": "kb-004",
        "category": "weather",
        "title": "폭우·천둥번개 시 대처",
        "content": (
            "산에서 천둥번개가 치면 즉시 낮은 곳으로 대피하세요. "
            "정상·능선·고립된 나무 근처는 매우 위험합니다. "
            "우산·등산스틱은 낮게 내리고, 큰 나무 아래는 피하세요. "
            "대피소·산장·암석 틈(물이 흐르지 않는 곳)으로 이동하세요. "
            "시간당 30mm 이상 강우 시 계곡은 급격히 불어나므로 계곡 근처 코스는 즉시 우회하세요."
        ),
        "keywords": ["폭우", "천둥", "번개", "비", "강우", "계곡", "날씨", "벼락"],
    },
    {
        "id": "kb-005",
        "category": "weather",
        "title": "강풍 대처",
        "content": (
            "초속 14m 이상 강풍 시 정상·능선 등 노출된 구간은 진행을 중단하세요. "
            "등산스틱을 활용해 균형을 잡고, 바위 모서리나 절벽 근처에서는 특히 주의하세요. "
            "배낭의 무게중심을 낮추고 바람이 약해질 때 이동하세요. "
            "악천후 예보 시 사전에 산행을 취소하는 것이 최선입니다."
        ),
        "keywords": ["강풍", "바람", "풍속", "돌풍"],
    },
    {
        "id": "kb-006",
        "category": "weather",
        "title": "안개·시야 불량 시 대처",
        "content": (
            "안개로 시야가 50m 이하로 제한될 경우 조난 위험이 높아집니다. "
            "GPS 앱으로 현재 위치를 수시로 확인하세요. "
            "능선 갈림길에서는 표지판을 꼼꼼히 확인하고, 리본 표식을 따라 이동하세요. "
            "안개가 짙어지면 하산 경로를 최단 거리로 변경하는 것을 권장합니다."
        ),
        "keywords": ["안개", "시야", "가시거리", "구름"],
    },
    {
        "id": "kb-007",
        "category": "wildfire",
        "title": "산불 발생 시 대처",
        "content": (
            "산불 발견 시 즉시 119 또는 산림청 산불신고 1688-9119에 신고하세요. "
            "연기가 이동하는 방향의 반대쪽, 즉 바람이 불어오는 방향으로 신속히 대피하세요. "
            "불길이 앞을 막고 있으면 이미 연소된 지역으로 대피하세요. "
            "코와 입을 젖은 수건이나 옷으로 막고 최대한 낮은 자세로 이동하세요. "
            "건조하고 바람이 강한 날은 산불위험예보를 반드시 확인하세요."
        ),
        "keywords": ["산불", "불", "화재", "연기", "산불위험", "불씨"],
    },
    {
        "id": "kb-008",
        "category": "landslide",
        "title": "산사태 위험 및 대처",
        "content": (
            "집중호우(시간당 20mm 이상) 시 산사태 위험이 급증합니다. "
            "산사태 징조: 갑작스러운 탁한 물 흐름, 나무 뿌리 꺾이는 소리, 땅 균열. "
            "즉시 계곡·급경사 지역을 벗어나 산 중턱의 견고한 지대로 수평 대피하세요. "
            "산사태 예측 경보 발령 시 즉시 하산하고 1688-9119에 신고하세요. "
            "국립공원 탐방로 통제 정보는 국립공원 홈페이지에서 실시간 확인하세요."
        ),
        "keywords": ["산사태", "토사", "급경사", "집중호우", "산사태경보", "토석류"],
    },
    {
        "id": "kb-009",
        "category": "safety",
        "title": "일몰 후 야간 산행 위험",
        "content": (
            "야간 산행은 낙상·조난 위험이 주간의 3배 이상입니다. "
            "일몰 시각 2~3시간 전에는 하산을 시작하세요. "
            "헤드랜턴은 반드시 여분 배터리와 함께 준비하고, 밝기 100루멘 이상을 사용하세요. "
            "국립공원 탐방로는 일몰 후 통제됩니다."
        ),
        "keywords": ["야간", "일몰", "해지다", "어둠", "저녁", "밤", "헤드랜턴"],
    },
    {
        "id": "kb-010",
        "category": "equipment",
        "title": "필수 등산 장비",
        "content": (
            "당일 산행 필수 장비: 등산화(발목 지지대), 등산스틱, 방수 재킷, 헤드랜턴, 비상식량·물(1인 1.5L 이상). "
            "개인 구급함: 탄력붕대, 지혈제, 소독약, 상처 덮개, 비상용 은박 담요. "
            "스마트폰 보조배터리(10000mAh 이상) 및 종이 지도. "
            "여름 산행에는 자외선 차단제(SPF 50 이상), 모자, 쿨링 타월을 추가하세요."
        ),
        "keywords": ["장비", "준비물", "챙기다", "배낭", "등산화", "헤드랜턴", "우비", "물품"],
    },
    {
        "id": "kb-011",
        "category": "equipment",
        "title": "겨울 산행 장비",
        "content": (
            "겨울 산행 필수 추가 장비: 아이젠(4발~10발), 스패츠(각반), 방한장갑(이중), 핫팩. "
            "체온 유지를 위해 미드레이어(플리스·다운)와 방풍·방수 아우터를 착용하세요. "
            "설상 산행 시 선글라스로 설맹(snow blindness)을 예방하세요. "
            "눈 덮인 탐방로에서는 평소 소요시간의 1.5~2배로 계획하세요."
        ),
        "keywords": ["겨울", "눈", "설산", "아이젠", "동계", "결빙", "빙판", "방한"],
    },
    {
        "id": "kb-012",
        "category": "course",
        "title": "초보자 산행 코스 선택",
        "content": (
            "초보자는 편도 거리 3km 이하, 소요 시간 2시간 이내 코스를 권장합니다. "
            "서울 근교 초보 추천: 인왕산(338m), 아차산(285m), 남산(262m), 청계산 옥녀봉 코스. "
            "전국 국립공원 초보 추천: 계룡산 동학사 코스, 내장산 케이블카 후 탐방로. "
            "탐방로 안내센터에서 현재 코스 통제 여부를 반드시 확인하세요."
        ),
        "keywords": ["초보", "입문", "쉬운", "초급", "처음", "가볍게", "쉽다", "간단"],
    },
    {
        "id": "kb-013",
        "category": "course",
        "title": "산행 계획 및 체력 관리",
        "content": (
            "본인 체력 수준보다 30% 쉬운 코스로 시작하세요. "
            "10~15분 등산 후 3~5분 휴식을 반복하는 리듬이 효과적입니다. "
            "무릎 통증 예방을 위해 하산 시 등산스틱을 적극 활용하고, 사선 하산을 권장합니다. "
            "당일 산행의 경우 10시 이전 출발, 14시 이전 정상 도착, 17시 이전 하산 완료를 목표로 하세요."
        ),
        "keywords": ["체력", "무릎", "하산", "피로", "속도", "페이스", "계획"],
    },
    {
        "id": "kb-014",
        "category": "course",
        "title": "어린이·노약자 동반 산행",
        "content": (
            "어린이·노약자 동반 시 성인 단독 산행 시간의 1.5~2배로 계획하세요. "
            "6세 이하 어린이의 하중 없는 최대 지속 도보 거리는 약 4km입니다. "
            "중간 쉼터와 화장실 위치를 사전에 확인하고, 아이가 지쳐 보이면 즉시 하산하세요. "
            "휠체어·유모차 접근 가능 탐방로: 국립공원 무장애 탐방로를 이용하세요."
        ),
        "keywords": ["어린이", "아이", "노약자", "가족", "노인", "유모차", "동반", "어린이동반"],
    },
    {
        "id": "kb-015",
        "category": "wildlife",
        "title": "산에서 야생동물 마주쳤을 때",
        "content": (
            "멧돼지: 등을 돌리고 달리지 말고 천천히 뒷걸음쳐 물러나세요. "
            "말벌: 검은색·향수 등 자극을 피하고, 발견 시 낮은 자세로 빠르게 자리를 피하세요. "
            "독사: 1~2m 거리를 유지하고 지팡이로 풀을 미리 헤쳐 가며 이동하세요. "
            "곰(설악산·지리산 등): 큰 소리로 말하며 천천히 후퇴하고, 절대 도망치지 마세요."
        ),
        "keywords": ["멧돼지", "말벌", "독사", "뱀", "곰", "야생동물", "벌", "동물"],
    },
    {
        "id": "kb-016",
        "category": "hydration",
        "title": "수분 보충 가이드",
        "content": (
            "성인 기준 30분마다 200ml 수분 섭취를 권장합니다. "
            "갈증을 느끼기 전에 미리 마시는 것이 중요하며, 땀을 많이 흘렸을 때는 스포츠음료로 전해질을 보충하세요. "
            "산에서 계곡물은 기생충·중금속 오염 가능성이 있으므로 그냥 마시지 마세요. "
            "무더운 날 당일 산행 시 성인 1인 최소 2L 물을 준비하세요."
        ),
        "keywords": ["수분", "물", "탈수", "음료", "계곡물", "더위", "갈증"],
    },
    {
        "id": "kb-017",
        "category": "safety",
        "title": "산행 신고·위치 공유",
        "content": (
            "산행 전 가족·지인에게 목적지, 출발시각, 예상 귀환 시각을 반드시 알리세요. "
            "등산로 입구 안전신고서를 작성하면 조난 시 구조 시간이 크게 단축됩니다. "
            "세이프 링크(Safe Link) 기능으로 실시간 위치를 보호자와 공유하면 더욱 안전합니다. "
            "스마트폰 배터리를 50% 이상 유지하고, 오프라인 지도를 미리 다운로드해 두세요."
        ),
        "keywords": ["신고", "위치", "공유", "보호자", "세이프링크", "안전신고", "배터리"],
    },
    {
        "id": "kb-018",
        "category": "national_park",
        "title": "국립공원 탐방로 통제 및 이용 규정",
        "content": (
            "국립공원 탐방로는 자연보호 및 안전을 위해 일부 구간이 계절별·기상 조건별로 통제됩니다. "
            "통제 구간 무단 진입 시 과태료(최대 50만원)가 부과됩니다. "
            "탐방로 통제 정보는 국립공원공단 공식 앱(Smart 국립공원)이나 홈페이지에서 확인하세요. "
            "산불 위험 시기(봄·건조기)에는 입산통제가 확대될 수 있습니다."
        ),
        "keywords": ["국립공원", "통제", "출입금지", "탐방로", "입산", "과태료", "제한"],
    },
    {
        "id": "kb-019",
        "category": "air_quality",
        "title": "미세먼지와 산행",
        "content": (
            "미세먼지 나쁨(PM2.5 36㎍/m³ 이상) 시 산행을 자제하고, 매우 나쁨(76㎍/m³ 이상)은 산행을 삼가세요. "
            "고강도 유산소 활동인 등산은 폐로 흡입하는 공기량이 일상의 3~5배로 증가합니다. "
            "국립산림과학원(NIFOS) 산림 미세먼지 관측시스템(aican.nifos.go.kr)에서 산림 지역 실시간 공기질을 확인하세요. "
            "KF94 마스크를 착용하면 미세먼지를 94% 차단할 수 있습니다."
        ),
        "keywords": ["미세먼지", "대기", "공기", "PM2.5", "마스크", "황사", "공기질"],
    },
    {
        "id": "kb-020",
        "category": "safety",
        "title": "등산로 길 찾기",
        "content": (
            "주요 갈림길과 정상에 설치된 이정표에는 거리(km)와 예상 소요시간이 표시됩니다. "
            "등산 앱(트랭글, 등산지도, 국가공간정보포털 Forest Trail)으로 현재 위치를 수시로 확인하세요. "
            "탐방로를 벗어났을 때는 당황하지 말고 지나온 경로를 되돌아가는 것이 가장 안전합니다. "
            "능선과 계곡의 방향을 파악하면 하산 방향을 추정할 수 있습니다."
        ),
        "keywords": ["길", "이정표", "방향", "지도", "GPS", "위치", "헤매다", "길을잃다"],
    },
    {
        "id": "kb-021",
        "category": "nutrition",
        "title": "산행 중 영양 보충",
        "content": (
            "1~2시간 산행마다 고열량 간식(에너지바, 견과류, 초콜릿)을 섭취하세요. "
            "탄수화물은 지속적 에너지 공급에 좋고, 단백질(육포, 치즈)은 근육 회복에 좋습니다. "
            "익히지 않은 산나물 채취·섭취는 독성 위험이 있으므로 삼가세요. "
            "고산 산행(1500m 이상) 시 고산증 예방을 위해 수분 섭취를 더욱 늘리세요."
        ),
        "keywords": ["간식", "음식", "에너지", "배고프다", "허기", "식량", "영양"],
    },
    {
        "id": "kb-022",
        "category": "season",
        "title": "봄 산행 주의사항",
        "content": (
            "봄(3~5월)은 건조하고 바람이 강해 산불 위험이 가장 높은 계절입니다. "
            "산불 위험 기간에는 입산 통제·화기 사용 금지가 광범위하게 시행됩니다. "
            "잔설이 남아 있는 구간은 아이젠이 필요하며, 그늘진 북사면은 4월에도 결빙 상태일 수 있습니다. "
            "산철쭉·진달래 철에는 탐방객이 급증하여 대중교통 이용을 권장합니다."
        ),
        "keywords": ["봄", "봄철", "4월", "5월", "진달래", "산불", "건조"],
    },
    {
        "id": "kb-023",
        "category": "season",
        "title": "여름 산행 주의사항",
        "content": (
            "여름(6~8월)은 폭우·집중호우로 인한 산사태 위험이 가장 높습니다. "
            "기상 예보를 당일 출발 전 반드시 확인하고, 폭염 특보 시 10~16시 산행을 피하세요. "
            "열사병 예방을 위해 목·팔목·이마에 찬 수건을 대어 체온을 낮추세요. "
            "자외선이 강하므로 자외선 차단제(SPF 50+)와 모자, 긴 소매를 착용하세요."
        ),
        "keywords": ["여름", "더위", "폭염", "장마", "열사병", "7월", "8월", "무덥다"],
    },
    {
        "id": "kb-024",
        "category": "season",
        "title": "가을 산행 주의사항",
        "content": (
            "가을(9~11월)은 단풍으로 탐방객이 폭증하는 성수기입니다. "
            "일몰이 빨라지므로 14시 이전 정상 도달을 목표로 하고, 산행 시간을 여름보다 단축하세요. "
            "낙엽이 쌓인 탐방로는 미끄러우므로 아이젠이나 미끄럼 방지 밑창 등산화를 착용하세요."
        ),
        "keywords": ["가을", "단풍", "10월", "11월", "성수기", "낙엽"],
    },
    {
        "id": "kb-025",
        "category": "season",
        "title": "겨울 산행 주의사항",
        "content": (
            "겨울(12~2월) 산행은 결빙·폭설로 인한 낙상과 저체온 위험이 큽니다. "
            "기온이 -10°C 이하이거나 강풍(-15°C 체감) 시 산행을 재고하세요. "
            "아이젠은 오름보다 내림에서 더 위험하므로 내림 시작 전에 꼭 착용하세요. "
            "해가 짧아 등산 계획을 여름보다 2~3시간 일찍 잡으세요."
        ),
        "keywords": ["겨울", "겨울산행", "결빙", "폭설", "동계", "12월", "1월", "2월", "추위"],
    },
    {
        "id": "kb-026",
        "category": "national_park",
        "title": "설악산 산행 정보",
        "content": (
            "설악산국립공원(1,708m)은 한국을 대표하는 산으로 울산바위, 대청봉, 공룡능선이 유명합니다. "
            "대청봉 정상 코스: 소공원~대청봉 7.5km, 편도 4~5시간 소요, 고난이도. "
            "주요 탐방로 거점: 소공원, 백담사, 오색, 한계령. "
            "겨울철 대청봉 코스는 12~3월 통제되는 경우가 많으며, 탐방 예약제가 시행됩니다."
        ),
        "keywords": ["설악산", "대청봉", "공룡능선", "울산바위", "소공원", "백담사"],
    },
    {
        "id": "kb-027",
        "category": "national_park",
        "title": "한라산 산행 정보",
        "content": (
            "한라산(1,950m)은 국내 최고봉으로 성판악·관음사 탐방로에서 백록담 정상에 오를 수 있습니다. "
            "성판악 코스(편도 9.6km, 4시간 30분)는 비교적 완만하고, 관음사 코스(8.7km)는 경사가 급합니다. "
            "입산 통제 시각: 성판악·관음사 12:00(동절기 기준). "
            "탐방 예약제 운영(한라산탐방예약시스템). 날씨 변화가 매우 빠르므로 기상 정보를 수시로 확인하세요."
        ),
        "keywords": ["한라산", "백록담", "성판악", "관음사", "제주"],
    },
    {
        "id": "kb-028",
        "category": "national_park",
        "title": "지리산 산행 정보",
        "content": (
            "지리산(1,915m)은 천왕봉이 최고점이며, 종주 코스는 성삼재~천왕봉 40km, 2박 3일 소요됩니다. "
            "대피소 예약제: 장터목, 세석, 연하천, 칠선대피소는 사전 예약 필수. "
            "천왕봉 당일 코스: 중산리 출발 편도 9km, 5~6시간 소요. "
            "하동포구~뱀사골~노고단은 가을 단풍 명소입니다."
        ),
        "keywords": ["지리산", "천왕봉", "종주", "성삼재", "대피소", "노고단"],
    },
    {
        "id": "kb-029",
        "category": "national_park",
        "title": "북한산 산행 정보",
        "content": (
            "북한산(836m)은 수도권에서 가장 접근성이 좋은 국립공원입니다. "
            "주요 코스: 백운대(우이동 출발 4.7km, 2~3시간), 비봉, 의상능선. "
            "탐방객 집중 시기(주말 가을)에는 탐방로별 입장 제한이 있습니다. "
            "도봉산 오봉·자운봉도 북한산국립공원에 속하며 다양한 초·중급 코스가 있습니다."
        ),
        "keywords": ["북한산", "백운대", "인수봉", "비봉", "도봉산", "우이동"],
    },
    {
        "id": "kb-030",
        "category": "nifos",
        "title": "국립산림과학원 산림 데이터 활용",
        "content": (
            "국립산림과학원(NIFOS)은 산림 안전과 관련된 다양한 실시간 관측 데이터를 제공합니다. "
            "산악기상시스템(mtweather.nifos.go.kr): 주요 산에 설치된 기상 관측소의 실시간 기온, 풍속, 강수량 제공. "
            "산림 미세먼지 관측시스템(aican.nifos.go.kr): 산림 지역 PM10·PM2.5 실시간 관측. "
            "산림과학지식서비스(know.nifos.go.kr): 산림 생태, 수종, 산림 연구 정보 제공. "
            "이 데이터를 활용해 산행 전 보다 정밀한 안전 판단이 가능합니다."
        ),
        "keywords": ["국립산림과학원", "NIFOS", "산악기상", "산림데이터", "미세먼지관측", "산림과학"],
    },
    {
        "id": "kb-031",
        "category": "environment",
        "title": "산림 보호 에티켓",
        "content": (
            "탐방로를 벗어난 훼손 행위는 생태계를 파괴합니다. 지정된 탐방로만 이용하세요. "
            "음식물 쓰레기는 절대 산에 버리지 말고 모두 가져가세요. "
            "야생동물에게 먹이를 주면 사람에 의존해 생태계를 교란합니다. "
            "취사·야영은 지정 구역에서만 허용됩니다. 무단 불 피우기는 산불 위험을 높입니다."
        ),
        "keywords": ["쓰레기", "환경", "자연", "취사", "야영", "캠핑", "에티켓", "쓰레기버리다"],
    },

    # ── NIFOS 산림과학 지식 (know.nifos.go.kr 기반) ─────────────────────────────
    {
        "id": "kb-032",
        "category": "nifos_carbon",
        "title": "산림 탄소흡수원과 등산객 역할",
        "content": (
            "국립산림과학원(NIFOS) 연구에 따르면 한국 산림은 연간 약 4,560만 톤의 CO₂를 흡수합니다. "
            "이는 국내 탄소 배출량의 약 6%에 해당하는 양입니다. "
            "등산객이 탐방로를 벗어나 식생을 훼손하면 탄소 흡수 능력이 저하됩니다. "
            "지정 탐방로 이용, 식물 채취 금지, 토양 훼손 방지가 산림 탄소 보전에 직결됩니다. "
            "광릉·지리산 등 플럭스 관측탑에서 실시간 탄소 흡수량이 모니터링됩니다."
        ),
        "keywords": ["탄소", "이산화탄소", "CO2", "탄소흡수", "탄소중립", "온실가스", "산림탄소"],
    },
    {
        "id": "kb-033",
        "category": "nifos_carbon",
        "title": "산림 탄소플럭스 관측 — NIFOS 생태플럭스 시스템",
        "content": (
            "국립산림과학원은 광릉(경기), 태안(충남), 양양(강원), 가리왕산(강원), 지리산(경남) 등에 "
            "에디 공분산(Eddy Covariance) 방식의 플럭스 관측탑을 운영합니다. "
            "NEE(순생태계교환량)가 음수(-) 이면 산림이 탄소를 흡수 중이고, "
            "양수(+) 이면 호흡에 의한 탄소 방출을 의미합니다. "
            "맑은 낮에는 광합성 활성화로 강한 탄소흡수(-5 μmol/m²/s 이하)가 나타납니다. "
            "이 데이터는 igportal.nifos.go.kr에서 실시간 확인할 수 있습니다."
        ),
        "keywords": ["플럭스", "탄소플럭스", "NEE", "광합성", "생태계", "igportal", "에디공분산"],
    },
    {
        "id": "kb-034",
        "category": "nifos_ecosystem",
        "title": "산림 생태계 건강성과 산행 안전",
        "content": (
            "건강한 산림 생태계는 사면 안정화, 토사 유출 억제, 홍수 완화 기능을 합니다. "
            "NIFOS 연구에서 뿌리 밀도가 높은 혼효림(침엽+활엽)은 산사태 저항성이 순림보다 1.3~1.8배 높습니다. "
            "산불·병해충으로 수목이 고사한 사면은 토사 결착력이 급감하여 집중호우 시 산사태 위험이 3~5배 증가합니다. "
            "산행 전 인근 산림 건강성 현황을 산림청 산림병해충 예보에서 확인하세요."
        ),
        "keywords": ["생태계", "산사태", "사면안정", "토사", "혼효림", "산림건강", "병해충"],
    },
    {
        "id": "kb-035",
        "category": "nifos_climate",
        "title": "기후변화와 한국 산림의 변화",
        "content": (
            "NIFOS 장기 모니터링에 따르면 한반도 평균기온이 100년간 1.8°C 상승하면서 "
            "식생 분포가 100m 이상 상향 이동했습니다. "
            "아고산대(해발 1,200~1,500m) 구상나무, 분비나무 등 냉온대 침엽수림이 급속 쇠퇴 중입니다. "
            "기온 상승으로 진드기(참진드기) 서식 고도가 해발 800m 이상으로 확대되었습니다. "
            "봄 개화 시기가 빨라져 고산 코스에서 이른 봄(3~4월) 잔설·결빙 구간이 여전히 위험합니다."
        ),
        "keywords": ["기후변화", "지구온난화", "구상나무", "아고산", "진드기", "SFTS", "식생변화"],
    },
    {
        "id": "kb-036",
        "category": "nifos_wildfire",
        "title": "NIFOS 산불 위험도 예측 시스템",
        "content": (
            "국립산림과학원은 기상청 데이터와 산림 연료(낙엽, 고사목) 정보를 결합한 "
            "산불위험지수(FWI) 예측 모델을 운영합니다. "
            "FWI 지수 51 이상 시 '매우 높음' 등급으로 입산 통제가 확대됩니다. "
            "건조한 봄철(3~5월)과 가을 건조기(11~12월)가 산불 위험 피크입니다. "
            "연소 속도는 풍속에 비례하여 초속 7m 바람에서 불이 1km를 약 8분 만에 이동할 수 있습니다. "
            "산불 위험 예보는 산림청 산불 재난 포털(forestfire.go.kr)에서 확인하세요."
        ),
        "keywords": ["산불위험지수", "FWI", "산불예측", "건조", "입산통제", "화재", "산불위험"],
    },
    {
        "id": "kb-037",
        "category": "nifos_landslide",
        "title": "NIFOS 산사태 위험지 분류와 대피 원칙",
        "content": (
            "국립산림과학원은 산사태 위험지를 1~5등급으로 분류합니다 (1등급=최고위험). "
            "경사 30° 이상, 토심 얕은 풍화토, 식생 훼손지가 고위험 조합입니다. "
            "시간당 강우량 20~30mm 이상에서 산사태 시작 확률이 급증하며 "
            "누적 강우량 150mm 이상이면 대규모 산사태 위험입니다. "
            "산사태 발생 시 계곡과 직각 방향(수평)으로 100m 이상 이동 후 고지대로 대피하세요. "
            "산사태 발생 이후 48시간 내 같은 지역에서 재발 가능성이 높습니다."
        ),
        "keywords": ["산사태위험지", "등급", "경사", "풍화토", "집중호우", "재발", "누적강우"],
    },
    {
        "id": "kb-038",
        "category": "nifos_insect",
        "title": "산림 병해충과 등산 주의사항",
        "content": (
            "참진드기(작은소참진드기)는 SFTS(중증열성혈소판감소증후군) 바이러스를 매개합니다. "
            "4~11월 활동 최성기이며 풀밭, 낙엽층에 서식합니다. "
            "예방: 긴 바지·긴 소매 착용, 기피제(DEET) 사용, 산행 후 2시간 내 전신 확인. "
            "진드기 발견 시 손으로 제거하지 말고 의료기관 방문 — 잘못 제거 시 머리 부분이 피부에 잔존합니다. "
            "털진드기 쓰쓰가무시증은 9~11월 가을 산행 후 발열 시 즉시 의원 방문 필요."
        ),
        "keywords": ["진드기", "SFTS", "쓰쓰가무시", "기피제", "풀밭", "털진드기", "벌레", "해충"],
    },
    {
        "id": "kb-039",
        "category": "nifos_ecosystem",
        "title": "산림 증발산과 여름 산행 체감온도",
        "content": (
            "NIFOS 플럭스 관측에 따르면 건강한 산림의 잠열플럭스(LE)는 여름 맑은 날 "
            "최대 400~600 W/m²에 달하며, 이는 주변 기온을 2~4°C 낮추는 냉각 효과를 냅니다. "
            "따라서 울창한 산림 내부는 개방된 능선보다 체감온도가 현저히 낮습니다. "
            "그러나 능선·정상부는 그늘이 없어 직달일사+복사열로 체감온도가 30°C 이상 올라갈 수 있습니다. "
            "여름 산행 시 정상 직전 구간에서 충분한 수분 보충과 휴식이 특히 필요합니다."
        ),
        "keywords": ["증발산", "잠열", "체감온도", "더위", "냉각", "능선", "여름산행", "열"],
    },
    {
        "id": "kb-040",
        "category": "nifos_data",
        "title": "NIFOS 실시간 산악기상 관측망 활용법",
        "content": (
            "국립산림과학원 산악기상시스템(mtweather.nifos.go.kr)은 주요 산 정상 및 능선에 "
            "설치된 관측소에서 기온, 풍속, 강수량, 습도를 10분 단위로 제공합니다. "
            "기상청 지상 관측과 달리 실제 산 높이에서 측정하므로 등산 계획에 더 적합합니다. "
            "고도 1,000m 기준 기온은 평지보다 약 6°C 낮고, 풍속은 2~3배 강하게 나타납니다. "
            "산 정상부 기상과 등산로 입구 기상의 차이가 크므로 산악 전용 기상 데이터를 반드시 확인하세요."
        ),
        "keywords": ["산악기상", "NIFOS", "관측소", "풍속", "기온차", "mtweather", "고도"],
    },
    {
        "id": "kb-041",
        "category": "nifos_carbon",
        "title": "산림 탄소 흡수와 등산객 행동 수칙",
        "content": (
            "산림 내 토양은 지상부 수목보다 2~3배 많은 탄소를 저장합니다. "
            "등산화 바닥이 토양을 다져 놓으면 토양 호흡이 억제되고 탄소 방출이 증가합니다. "
            "NIFOS 연구에 따르면 탐방로 1m 외 지역의 토양탄소는 탐방로 중심부보다 1.4배 높습니다. "
            "따라서 지정 탐방로에서 벗어나지 않는 것이 탄소 보전과 생태 보호 모두에 중요합니다. "
            "낙엽을 밟지 않으려는 과도한 좌우 이탈은 오히려 식생을 훼손합니다."
        ),
        "keywords": ["토양탄소", "탐방로", "토양", "탄소저장", "식생훼손", "탄소보전"],
    },
    {
        "id": "kb-042",
        "category": "nifos_ecosystem",
        "title": "산림 생물다양성과 보호종 마주쳤을 때",
        "content": (
            "국립산림과학원 조사에 따르면 국내 산림에는 식물 4,500여 종, 곤충 15,000여 종이 서식합니다. "
            "반달가슴곰(지리산), 산양(설악산·태백산), 수리부엉이 등 멸종위기종 서식지에서는 "
            "소음·조명 자제, 접근 금지 구역 준수가 법적 의무입니다. "
            "보호종 발견 시 국립공원공단(1588-0820)이나 지역 산림청에 제보하세요. "
            "식물 채취는 지정 구역 외에서 전면 금지이며 위반 시 50만 원 이하 과태료가 부과됩니다."
        ),
        "keywords": ["생물다양성", "반달곰", "산양", "멸종위기", "보호종", "채취금지", "생태"],
    },
    {
        "id": "kb-043",
        "category": "nifos_soil",
        "title": "등산로 토양 훼손과 침식 방지",
        "content": (
            "NIFOS 연구에 따르면 연간 탐방객 10만 명 이상 구간의 탐방로 폭은 10년간 평균 2.3배 넓어집니다. "
            "탐방로 침식이 심한 구간에서 낙석·미끄러짐 사고가 집중 발생합니다. "
            "우천 시 물이 고인 탐방로를 피해 옆으로 돌아가면 침식이 가속됩니다 — 물 위를 그냥 걷는 것이 낫습니다. "
            "계단·데크가 없는 급사면에서 지그재그로 내려오면 탐방로 확장을 방지할 수 있습니다. "
            "국립공원 탐방로 복원 사업은 생태탐방원 및 국립공원공단 봉사 프로그램으로 참여 가능합니다."
        ),
        "keywords": ["토양침식", "탐방로훼손", "미끄러짐", "낙석", "탐방로복원", "데크"],
    },
    {
        "id": "kb-044",
        "category": "nifos_air",
        "title": "산림 미세먼지 정화 능력과 등산",
        "content": (
            "NIFOS 연구에 따르면 산림 1ha는 연간 PM10 약 46kg, PM2.5 약 10kg을 흡착·제거합니다. "
            "산림 내 PM2.5 농도는 도심보다 평균 20~40% 낮게 유지됩니다. "
            "그러나 강풍이 부는 날에는 도심의 미세먼지가 산 사면을 타고 능선까지 유입됩니다. "
            "미세먼지 '나쁨' 이상 시에도 울창한 계곡 코스는 능선보다 오염도가 낮은 편입니다. "
            "NIFOS 산림 미세먼지 실시간 데이터(aican.nifos.go.kr)에서 관측소별 현황을 확인하세요."
        ),
        "keywords": ["미세먼지정화", "PM10", "PM2.5", "산림공기", "대기질", "aican", "능선"],
    },
    {
        "id": "kb-045",
        "category": "nifos_wildfire",
        "title": "산불 발생 후 산행 재개 시 주의사항",
        "content": (
            "산불 발생 후 6개월 이내 해당 지역 산행은 매우 위험합니다. "
            "고사목 낙지(죽은 나뭇가지)가 바람에 쉽게 떨어져 직접 타격 사고가 발생합니다. "
            "지표 식생이 소실된 사면은 강우 시 표면 흐름이 급증하여 침식·소규모 토사류가 발생합니다. "
            "NIFOS 산불피해지 모니터링에 따르면 초기 식생 회복에 3~5년, 완전 회복에 30~50년이 소요됩니다. "
            "산불 피해지 출입 제한은 산림청 산불재난포털에서 최신 정보를 확인하세요."
        ),
        "keywords": ["산불피해", "고사목", "낙지", "식생회복", "출입제한", "피해지", "산불후"],
    },
    {
        "id": "kb-046",
        "category": "nifos_ecosystem",
        "title": "고산 습지와 물이끼 지대 보호",
        "content": (
            "해발 1,000m 이상 고산 습지(덕유산, 오대산 등)는 식물다양성의 보고이자 "
            "산림 수원함양 기능의 핵심입니다. "
            "물이끼(Sphagnum) 군락지는 자체 무게의 20배까지 물을 머금어 건기 수량을 유지합니다. "
            "국립공원 습지 보전지구 내 탐방로 이탈은 엄격히 금지됩니다. "
            "습지 주변 탐방로는 데크 보행로가 설치된 경우 반드시 데크 위로만 이동하세요. "
            "NIFOS는 고산 습지 탄소 저장량이 일반 산림의 5~10배에 달한다고 보고합니다."
        ),
        "keywords": ["고산습지", "물이끼", "습지", "수원함양", "덕유산", "오대산", "습원"],
    },
]


# ── 검색 엔진 ──────────────────────────────────────────────────────────────────
import math as _math

def _tokenize(text: str) -> set:
    """Korean/English text tokenizer: word tokens + bigrams."""
    clean = re.sub(r"[^가-힣a-z0-9\s]", " ", text.lower())
    words = set(w for w in clean.split() if len(w) >= 2)
    chars = re.sub(r"\s+", "", clean)
    bigrams = {chars[i: i + 2] for i in range(len(chars) - 1)} if len(chars) > 1 else set()
    return words | bigrams


# 지식베이스(47개) + 산 정보 DB(최대 500개) 문서가 다 들어가도록 여유 있게 잡는다
@lru_cache(maxsize=1024)
def _cached_tokenize(text: str) -> frozenset:
    return frozenset(_tokenize(text))


# ── IDF 사전 (모듈 임포트 시 1회 사전계산 → 요청마다 재계산 없음) ─────────────

def _build_idf() -> dict:
    N = len(KNOWLEDGE_BASE)
    df: dict = {}
    for doc in KNOWLEDGE_BASE:
        doc_text = f"{doc['title']} {doc['content']} {' '.join(doc['keywords'])}"
        for token in _cached_tokenize(doc_text):
            df[token] = df.get(token, 0) + 1
    return {token: _math.log((N + 1) / (cnt + 1)) + 1 for token, cnt in df.items()}

_IDF: dict = _build_idf()  # 모듈 로드 시 1회만 실행


def _get_idf(token: str) -> float:
    return _IDF.get(token, 1.0)


def _weighted_score(query_tokens: frozenset, doc_tokens: frozenset) -> float:
    """IDF 가중 점수: 드문 단어 일치에 높은 가중치."""
    if not query_tokens:
        return 0.0
    matched = query_tokens & doc_tokens
    if not matched:
        return 0.0
    idf_sum = sum(_get_idf(t) for t in matched)
    return idf_sum / (len(query_tokens) + 0.5)


def retrieve_from_knowledge_base(query: str, top_k: int = 3) -> list:
    """정적 지식 베이스에서 관련 문서 검색 (IDF 가중 + 키워드 직접 매칭 보너스)."""
    query_tokens = _cached_tokenize(query)
    query_lower = query.lower()
    scored = []
    for doc in KNOWLEDGE_BASE:
        doc_text = f"{doc['title']} {doc['content']} {' '.join(doc['keywords'])}"
        doc_tokens = _cached_tokenize(doc_text)
        score = _weighted_score(query_tokens, doc_tokens)
        if score > 0:
            # keywords 리스트에 쿼리가 직접 포함된 단어마다 +0.4 보너스
            kw_bonus = sum(0.4 for kw in doc["keywords"] if kw in query_lower)
            scored.append((score + min(kw_bonus, 1.2), doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def _course_search_text(course: dict) -> str:
    return " ".join([
        course.get("mountain", ""),
        course.get("name", ""),
        course.get("region", ""),
        " ".join(course.get("highlights", []) or []),
    ])


@lru_cache(maxsize=1)
def _trail_search_index():
    """탐방로 역색인 (토큰 → 코스 번호들). 워커당 한 번만 만든다.

    예전에는 메시지마다 8천여 개 코스 문장을 전부 다시 토큰화했다.
    """
    from .loaders import load_public_trail_courses
    courses = load_public_trail_courses()
    postings: dict = {}
    by_mountain: dict = {}
    for index, course in enumerate(courses):
        for token in _tokenize(_course_search_text(course)):
            postings.setdefault(token, array("I")).append(index)
        by_mountain.setdefault(course.get("mountain", ""), array("I")).append(index)
    return courses, postings, by_mountain


def retrieve_trail_context(query: str, mountain_name: str = "", top_k: int = 3) -> list:
    """국립공원 탐방로 데이터에서 관련 코스 검색."""
    courses, postings, by_mountain = _trail_search_index()

    search_text = f"{query} {mountain_name}".strip()
    query_tokens = _cached_tokenize(search_text)

    # _weighted_score와 같은 점수: 일치 토큰 IDF 합 / (질문 토큰 수 + 0.5)
    idf_sums: dict = {}
    for token in query_tokens:
        indexes = postings.get(token)
        if not indexes:
            continue
        weight = _get_idf(token)
        for index in indexes:
            idf_sums[index] = idf_sums.get(index, 0.0) + weight
    denominator = len(query_tokens) + 0.5
    scores = {index: total / denominator for index, total in idf_sums.items()}
    # 산 이름 직접 일치 시 강하게 부스팅
    if mountain_name:
        for mountain, indexes in by_mountain.items():
            if mountain_name in mountain:
                for index in indexes:
                    scores[index] = scores.get(index, 0.0) + 1.5

    # 동점이면 원래 코스 순서를 유지(이전 구현과 같은 결과)
    ranked = sorted(
        (index for index, score in scores.items() if score > 0),
        key=lambda index: (-scores[index], index),
    )
    results = [(scores[index], courses[index]) for index in ranked[:top_k]]
    passages = []
    for _, course in results[:top_k]:
        diff = {"easy": "초급", "medium": "중급", "hard": "고급"}.get(course.get("difficulty", ""), "")
        highlights = "  ".join(course.get("highlights", []) or [])
        passages.append(
            f"[탐방로] {course.get('mountain', '')} / {course.get('name', '')}: "
            f"거리 {course.get('distance_km', '-')}km, 소요 {course.get('duration_min', '-')}분, 난이도 {diff}. "
            f"{highlights}"
        )
    return passages


def retrieve_disaster_context(mountain_name: str, top_k: int = 3) -> list:
    """국립공원 재난위험지구 데이터에서 해당 산 위험 지구 검색."""
    from .loaders import load_disaster_risk_zones, normalize_search_text
    if not mountain_name:
        return []

    zones = load_disaster_risk_zones()
    needle = normalize_search_text(mountain_name)
    passages = []
    for zone in zones:
        if needle in zone.get("search_text", ""):
            factor = zone.get("risk_factor", "재난위험지구")
            location = zone.get("location") or zone.get("district", "")
            evacuation = zone.get("evacuation_place", "")
            passages.append(
                f"[재난위험] {location}: 위험요인 '{factor}'. "
                + (f"대피장소: {evacuation}" if evacuation else "")
            )
        if len(passages) >= top_k:
            break
    return passages


def retrieve_mountain_knowledge(query: str, mountain_name: str = "", top_k: int = 3) -> list:
    """DB의 MountainKnowledge에서 관련 산 정보 검색."""
    try:
        from .models import MountainKnowledge
    except Exception:
        return []

    qs = MountainKnowledge.objects.filter(summary__gt="")
    if mountain_name:
        # 해당 산 정보를 우선 포함
        qs = qs.filter(mountain_name__icontains=mountain_name.replace(" ", ""))

    query_tokens = _cached_tokenize(query)
    scored = []
    for obj in qs[:500]:
        text = f"{obj.mountain_name} {obj.summary} {obj.detail} {obj.selection_reason}"
        doc_tokens = _cached_tokenize(text)
        score = _weighted_score(query_tokens, doc_tokens)
        if mountain_name and mountain_name in obj.mountain_name:
            score += 1.5
        if score > 0:
            scored.append((score, obj))

    scored.sort(key=lambda x: x[0], reverse=True)
    passages = []
    for _, obj in scored[:top_k]:
        body = obj.summary[:500] or obj.detail[:500]
        reason = f"  선정이유: {obj.selection_reason[:150]}" if obj.selection_reason else ""
        height = f" ({obj.height_m}m)" if obj.height_m else ""
        passages.append(
            f"[산 정보] {obj.mountain_name}{height} / {obj.region}: {body}{reason}"
        )
    return passages


def build_rag_context(query: str, mountain_name: str = "") -> str:
    """사용자 질문에 대한 RAG 컨텍스트 문자열 생성.

    지식 베이스 + 탐방로 데이터 + 재난위험 데이터를 검색하여 통합 컨텍스트를 반환합니다.
    """
    sections = []

    kb_docs = retrieve_from_knowledge_base(query, top_k=3)
    if kb_docs:
        kb_lines = "\n".join(
            f"• [{doc['title']}] {doc['content'][:400]}" for doc in kb_docs
        )
        sections.append(f"[산림 안전 지식 - 출처: NIFOS/국립공원공단]\n{kb_lines}")

    mk_passages = retrieve_mountain_knowledge(query, mountain_name, top_k=3)
    if mk_passages:
        sections.append("[산 상세 정보 - 출처: 산림청]\n" + "\n".join(mk_passages))

    trail_passages = retrieve_trail_context(query, mountain_name, top_k=3)
    if trail_passages:
        sections.append("[관련 탐방로 정보 - 출처: 국립공원공단]\n" + "\n".join(trail_passages))

    if mountain_name:
        disaster_passages = retrieve_disaster_context(mountain_name, top_k=3)
        if disaster_passages:
            sections.append("[재난위험 지구 - 출처: 국립공원공단]\n" + "\n".join(disaster_passages))

    return "\n\n".join(sections)
