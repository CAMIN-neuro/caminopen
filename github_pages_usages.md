Github Pages 사용법
===================

이번 문서에서는 크게 1. HTML 페이지를 만드는 과정 2. HTML 페이지를 배포하는 과정 으로 나눠서 진행하겠습니다. 먼저 HTML 페이지를 만드는 여러가지 툴이 있지만 여기서는 Shpinx 를 이용하겠습니다. 다음 명령어를 통해, 프로그램을 설치할 수 있습니다.

    $ pip install Sphinx

## 1. HTML 페이지 만드는 과정
### 1.1 sphinx-quickstart

먼저 이후 생성되는 파일들을 한곳에 모아둘 root 디렉토리(docs)를 생성합니다. 그후 다음 명령어를 통해 기본 문서를 만듭니다. 여기서 소스코드가 들어가는 "source" 폴더와 html로 변환된 파일이 들어갈 "build" 폴더를 분리할지 합칠지 결정해야 합니다. 이번에는 별도로 관리하기 위해 y 를 입력하겠습니다.

<p align="center"><img src="https://github.com/CAMIN-neuro/caminopen/blob/master/figure/pages/sphinx-quickstart_1.png" width=800 height=500></p>

그리고 다음과 같이 프로젝트명, 저자, 버전을 입력합니다. 추후에 *conf.py* 파일에서 수정가능하니, 임시로 적어도 괜찮습니다.

<p align="center"><img src="https://github.com/CAMIN-neuro/caminopen/blob/master/figure/pages/sphinx-quickstart_2.png" width=800 height=500></p>

다음으로 언어를 선택해주는데, 여기서는 en 으로 설정하겠습니다.

<p align="center"><img src="https://github.com/CAMIN-neuro/caminopen/blob/master/figure/pages/sphinx-quickstart_3.png" width=800 height=500></p>

그러면 다음과 같이 root 폴더에 파일들이 생기는것을 볼 수 있습니다.

<p align="center"><img src="https://github.com/CAMIN-neuro/caminopen/blob/master/figure/pages/sphinx-quickstart_tree.png" width=800 height=500></p>

- build 폴더는 문서화 시킨 결과물이 담기는 곳
- make.bat, Makefile 파일은 각각 윈도우, 유닉스 계열에서 빌드 할때 사용하는 파일
- source 폴더는 소스 코드를 담아두는 곳
- conf.py 파일은 환경설정 하는 파일
- index.rst 파일은 문서의 첫페이지
- _static, _templates 폴더는 이미지같이 외부 자료를 저장하는 곳

### 1.2 빌드 (HTML 생성)
다음 명령어를 통해 html 문서를 만들수 있습니다.

<p align="center"><img src="https://github.com/CAMIN-neuro/caminopen/blob/master/figure/pages/build.png" width=800 height=500></p>

만들어진 문서는 *build/html/index.html* 파일을 열어보면 다음과 같이 기본 홈페이지가 만들어진것을 확인할 수 있습니다.

<p align="center"><img src="https://github.com/CAMIN-neuro/caminopen/blob/master/figure/pages/build-html.png" width=800 height=500></p>

### 1.3 테마 설정
기본적으로 스핑크스에서는 여러가지 테마들을 선택하여 보다 빠르게 꾸밀수 있습니다. https://sphinx-themes.org/ 에 들어가면 여러가지 테마들중 하나를 선택할 수 있습니다. 이번에는 가장 보편적인 Read the Docs 를 이용하겠습니다. 

<p align="center"><img src="https://github.com/CAMIN-neuro/caminopen/blob/master/figure/pages/theme_org.png" width=800 height=500></p>

공식문서에 나와 있는대로, 다음 명령어를 입력하고
    $ pip install sphinx-rtd-theme
    
*source/conf.py*에 html_theme = 'sphinx_rtd_theme' 변수를 설정해주고, 다시 빌드해주면 다음과 같이 테마가 변경된것을 확인할 수 있습니다.
    
<p align="center"><img src="https://github.com/CAMIN-neuro/caminopen/blob/master/figure/pages/theme-build.png" width=800 height=500></p>
    
### 1.4 문서 작성하기
sphinx는 기본적으로 rst(reStructuredText) 문서만 인식하지만, md(markdown) 같은 다른 형식으로 작성한 파일도 conf.py에서 확장자만 인식하게 해준다면 사용가능 합니다 (CommonMarkParser 이용). 

rst 문법을 참고하고 싶으시면, 다음 두 링크를 보시면 됩니다. 첫번째는 핵심만 담겨져 있는것이고, 두번째는 보다 자세한 내용으로 참고하시면 될것 같습니다.
- https://itholic.github.io/etc-rst/
- https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html


## 2. HTML 페이지를 배포하는 과정
### 2.1 첫 페이지 연결
지금 까지 작성된 문서는 root 디렉토리(docs)에 저장되어 있습니다. 해당 폴더를 최종적으로 배포할 코드가 들어있는 폴더로 이동해야 합니다. 실제 배포할때는 깃허브 사용법에 적혀있는 clone 된 폴더에 옮겨야 하지만, 이번에는 **test** 라는 폴더에 **docs** 와 **code** 파일이 존재하는 상황으로 진행하겠습니다. 다음으로 docs 폴더에 **index.html** 파일을 다음과 같이 생성합니다.

    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=./build/html/index.html" />
        </head>
    </html>
    
해당 파일을 만드는 이유는 외부에서 접속할때, 이전에 만든 문서의 첫페이지로 가도록 설정하기 위함 때문입니다.

### 2.2 테마 적용
기본적으로 github pages는 jekyll 기반으로 만들어지기 때문에, 위에서 적용한 테마가 배포할때 적용되지 않습니다. 따라서 docs 폴더에 .nojekyll 라는 빈 파일을 생성하여, jekyll 방식이 아닌 위에서 적용한 테마가 적용되도록 만듭니다.

### 2.3 깃허브 세팅
배포할려는 파일들을 모두 github에 올리고 해당 레파지토리로 이동하시면, *Settings -> Pages*로 이동하시면 다음과같은 페이지를 볼 수 있습니다.

<p align="center"><img src="https://github.com/CAMIN-neuro/caminopen/blob/master/figure/pages/page.pngs" width=800 height=500></p>
    
원하는 Branch를 선택한후 다음과 같이 오른쪽 폴더를 선택하여 **/docs** 를 선택하고 Save를 눌러주시면 됩니다.
    
<p align="center"><img src="https://github.com/CAMIN-neuro/caminopen/blob/master/figure/pages/branch_docs.png" width=800 height=500></p>

약간의 시간이 지나면, 다음과 같이 주소가 나타나는것을 보실 수 있습니다.

<p align="center"><img src="https://github.com/CAMIN-neuro/caminopen/blob/master/figure/pages/pages_site.png" width=800 height=500></p>

해당 링크를 클릭하면 다음과 같이 로컬에서 만든 페이지와 동일한 페이지가 배포 되었음을 확인할 수 있습니다. 따라서 홈페이지를 배포할 때는 위에서 받은 링크를 배포하시면 됩니다.

## 3. 번외
### 3.1 View page source -> Edit on Github 로 변경
첫페이지 오른쪽 상단에 있는 View page source를 Edit on Github로 바꿔 소스코드를 볼때 해당 홈페이지가 아니라, 깃허브에 있는 코드로 이동하게 만들어 주는 버튼입니다. docs/source/conf.py 파일에 # -- Options for HTML output 아래에 다음 코드를 입력하면 됩니다.

    html_show_sourcelink = True
    html_context = {
        'display_github': True,
        'github_user': 'CAMIN-neuro',
        'github_repo': '레파지토리 이름',
        'github_version': '브랜치명/docs/source/',
    }

