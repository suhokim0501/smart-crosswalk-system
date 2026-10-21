# 사용자가 직접 GitHub에 올리기

이 프로젝트는 로컬에서만 준비했습니다. 자동 로그인, 원격 저장소 생성, 업로드는 수행하지 않습니다.

## 1. 웹에서 빈 저장소 생성

1. GitHub에서 본인 계정으로 로그인합니다.
2. **New repository**를 선택하고 이름을 `smart-crosswalk-system`으로 입력합니다.
3. 원하는 공개 범위를 선택합니다.
4. **README, .gitignore, License를 추가하는 옵션은 선택하지 않습니다.** 로컬 이력과 바로 연결할 빈 저장소를 만듭니다.
5. **Create repository**를 누릅니다.

## 2. 로컬 저장소 연결과 업로드

아래 `<YOUR_USERNAME>`을 실제 GitHub 사용자 이름으로 바꾸고 터미널에서 실행합니다.
인증은 GitHub가 안내하는 방식으로 직접 진행하면 됩니다.

```bash
cd /Users/suho/Career/smart-crosswalk-system
git status
git remote add origin https://github.com/<YOUR_USERNAME>/smart-crosswalk-system.git
git push -u origin main
```

이미 `origin`이 등록되어 있다면 `git remote -v`로 확인한 뒤, 필요할 때 직접 주소를 변경합니다.

```bash
git remote set-url origin https://github.com/<YOUR_USERNAME>/smart-crosswalk-system.git
```

`models/*.pt`, `data/videos/*.mp4`, `outputs/`, 가상 환경은 `.gitignore`에 의해 제외됩니다.
로컬 모델과 영상을 새로 넣더라도 기본 `git add .` 대상에 들어가지 않습니다.

업로드 후 웹에서 README가 표시되는지 확인합니다. GitHub Actions가 활성화되어 있다면
**Actions → Core tests**에서 테스트 결과를 확인할 수 있습니다.
