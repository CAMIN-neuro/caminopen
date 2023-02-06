Github Pages 사용법
===================

이번 문서에서는 크게 1. HTML 페이지를 만드는 과정 2. HTML 페이지를 배포하는 과정 으로 나눠서 진행하겠습니다. 먼저 HTML 페이지를 만드는 여러가지 툴이 있지만 여기서는 Shpinx 를 이용하겠습니다. 다음 명령어를 통해, 프로그램을 설치할 수 있습니다.

    $ pip install Sphinx

## 1. HTML 페이지 만드는 과정
### 1.1 sphinx-quickstart

먼저 이후 생성되는 파일들을 한곳에 모아둘 root 디렉토리를 생성합니다. 그후 다음 명령어를 통해 기본 문서를 만듭니다. 여기서 소스코드가 들어가는 "source" 폴더와 html로 변환된 파일이 들어갈 "build" 폴더를 분리할지 합칠지 결정해야 합니다. 이번에는 별도로 관리하기 위해 y 를 입력하겠습니다.

<p align="center"><img src="sphinx-quickstart" width=800 height=500></p>

그리고 다음과 같이 프로젝트명, 저자, 버전을 입력합니다. 추후에 *conf.py* 파일에서 수정가능하니, 임시로 적어도 괜찮습니다.

<p align="center"><img src="sphinx-quickstart" width=800 height=500></p>

다음으로 언어를 선택해주는데, 여기서는 en 으로 설정하겠습니다.

<p align="center"><img src="sphinx-quickstart" width=800 height=500></p>

그러면 다음과 같이 root 폴더에 파일들이 생기는것을 볼 수 있습니다.

<p align="center"><img src="sphinx-quickstart_tree" width=800 height=500></p>

- build 폴더는 문서화 시킨 결과물이 담기는 곳
- make.bat, Makefile 파일은 각각 윈도우, 유닉스 계열에서 빌드 할때 사용하는 파일
- source 폴더는 소스 코드를 담아두는 곳
- conf.py 파일은 환경설정 하는 파일
- index.rst 파일은 문서의 첫페이지
- _static, _templates 폴더는 이미지같이 외부 자료를 저장하는 곳

### 1.2 빌드 (HTML 생성)
다음 명령어를 통해 html 문서를 만들수 있습니다.

<p align="center"><img src="build" width=800 height=500></p>

만들어진 문서는 *build/html/index.html* 파일을 열어보면 다음과 같이 기본 홈페이지가 만들어진것을 확인할 수 있습니다.

<p align="center"><img src="build->html" width=800 height=500></p>

### 1.3 테마 설정
기본적으로 스핑크스에서는 여러가지 테마들을 선택하여 보다 빠르게 꾸밀수 있습니다. https://sphinx-themes.org/ 에 들어가면 여러가지 테마들중 하나를 선택할 수 있습니다. 이번에는 가장 보편적인 Read the Docs 를 이용하겠습니다. 

<p align="center"><img src="theme_org" width=800 height=500></p>

공식문서에 나와 있는대로, 다음 명령어를 입력하고
    $ pip install sphinx-rtd-theme
    
*source/conf.py*에 html_theme = 'sphinx_rtd_theme' 변수를 설정해주고, 다시 빌드해주면 다음과 같이 테마가 변경된것을 확인할 수 있습니다.
    
<p align="center"><img src="theme->build" width=800 height=500></p>
    
### 1.4 문서 작성하기
sphinx는 기본적으로 rst(reStructuredText) 문서만 인식하지만, md(markdown) 같은 다른 형식으로 작성한 파일도 conf.py에서 확장자만 인식하게 해준다면 사용가능 합니다 (CommonMarkParser 이용). 이번에는 rst 문서만 이용하여 진행하겠습니다.
    
    
    
    
    
    
    
