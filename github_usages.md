깃허브 사용법
=============

## 1. git clone
해당 명령어는 기존에 있는 프로젝트를 local로 가져오는 명령어로, 변경된 데이터를 깃허브에 올리기전에 항상 실행해야 합니다. 
<p align="center"><img src="https://github.com/CAMIN-neuro/caminopen/blob/master/figure/github/clone.png" width=100 height=50></p>

url은 원하는 레파지토리의 해당위치에서 복사할 수 있습니다.
<p align="center"><img src="github_url.png" width="가로 사이즈" height="세로 사이즈"></p>

## 2. git add .
먼저 git status 라는 명령어는 git의 상태를 나타내는 명령어 입니다. 추가할 파일(여기서는 new_file) 해당 폴더로 옮기고, git status 명령어를 치면 다음과 같습니다.
<p align="center"><img src="git_add_status.png" width="가로 사이즈" height="세로 사이즈"></p>

추가된 파일이 **Untracked files** 에 있는걸 볼수있습니다. 해당파일을 commit 하기전 저장하는 용도로 해당 명령어를 사용합니다.
<p align="center"><img src="git_add.png" width="가로 사이즈" height="세로 사이즈"></p>

## 3. git commit
해당 명령어는 git이 파일을 추적하도록 만드는 명령어로, 앞서 add 된 파일에만 적용할 수 있습니다.
<p align="center"><img src="git_commit.png" width="가로 사이즈" height="세로 사이즈"></p>

- 주의할점은 맨처음에 올리는 파일은 -m "내용" 을 붙여 올려야 가능합니다.

## 4. git push origin branch
git commit 까지 완료했다면 local에 있는 git은 업데이트가 되었지만, github에 있는 git은 아직 업데이트가 안되어 있습니다. 따라서 해당 명령어를 이용하여 github에 업데이트를 진행합니다. 아이디는 계정명을 입력하면 되고, 토큰은 깃허브 설정에서 발급받을 수 있습니다.
<p align="center"><img src="git_push.png" width="가로 사이즈" height="세로 사이즈"></p>

branch는 다음부분에서 확인할 수 있습니다. 
<p align="center"><img src="git_push_branch.png" width="가로 사이즈" height="세로 사이즈"></p>

## 5. 부가적인 명령어
### 5.1 git log
해당 명령어는 여태까지 추적된 git을 나타내주는 함수입니다.
<p align="center"><img src="git_log.png" width="가로 사이즈" height="세로 사이즈"></p>

### 5.2 git reset
해당 명령어는 github에 올리고 다시 돌릴때 사용하는 명령어 입니다. 숫자는 git log상 돌아가고 싶은 위치의 7자리를 넣으면 됩니다. (한번 되돌리면 그 이후의 내용은 복구가 안되니, 신중하게 사용해주세요.)
<p align="center"><img src="git_reset.png" width="가로 사이즈" height="세로 사이즈"></p>
