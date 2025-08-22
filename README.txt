CMAPSS 터보팬 엔진 데이터셋 요약 (Korean README)
데이터셋 구성
데이터셋    Train개수	  Test 개수   조건(Conditions)	   고장 모드(Fault Modes)
FD001       100	        100         ONE (Sea Level)	    ONE (HPC Degradation)
FD002	    260	        259	        SIX	                ONE (HPC Degradation)
FD003	    100	        100	        ONE (Sea Level)	    TWO (HPC Degradation, Fan Degradation)
FD004	    248	        249	        SIX	                TWO (HPC Degradation, Fan Degradation)
실험 시나리오 (Experimental Scenario)

각 데이터셋은 다변량(multivariate) 시계열의 집합입니다.

각 시계열은 서로 다른 엔진(unit) 에서 수집되며, 같은 타입의 엔진 플릿(fleet)으로 간주할 수 있습니다.

각 엔진은 서로 다른 초기 마모와 제조 편차를 가지고 시작하며, 이는 정상 범위로 간주됩니다(즉, 고장 아님).

엔진 성능에 큰 영향을 주는 3개의 운영 조건(operational settings) 이 포함되어 있으며, 센서 노이즈가 존재합니다.

훈련(Train) 시계열: 시작은 정상 상태이며, 어느 시점부터 고장이 발생해 고장(시스템 실패)에 도달할 때까지 진행됩니다.

테스트(Test) 시계열: 고장에 도달하기 일정 시간 이전에서 종료됩니다.

목표: 테스트 세트의 각 시계열에 대해 고장까지 남은 운전 사이클 수(RUL, Remaining Useful Life) 를 예측하는 것입니다. (테스트 데이터에 대한 정답 RUL 벡터가 제공됩니다.)

파일 형식 및 컬럼 설명

데이터는 공백으로 구분된 26개 열의 텍스트 파일(zip 압축)로 제공됩니다.

각 행은 단일 운전 사이클에서의 스냅샷이며, 각 열은 다른 변수를 나타냅니다.

열 구성(총 26열):

unit 번호

time(사이클)

operational setting 1

operational setting 2

operational setting 3

~ 26) 센서 측정값 21개 (즉, sensor_1 ~ sensor_21)

참고: 인덱스 6~26까지가 센서 값이므로 총 21개 센서입니다. (열 번호 26은 센서 21번째에 해당)

참고 문헌

A. Saxena, K. Goebel, D. Simon, and N. Eklund,
“Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,”
Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO, Oct 2008.