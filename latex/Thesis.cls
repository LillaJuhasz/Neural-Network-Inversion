%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Szakdolgozat/Diplomamunka/TDK dolgozat minta
% Miskolci Egyetem
% Informatikai Intezet
%
% 
%
%
% Steve R. Gunn Thesis templete--je alapjan.
%
% Original header:
%% This is file `Thesis.cls', based on 'ECSthesis.cls', by Steve R. Gunn
%% generated with the docstrip utility.
%%
%% Created by Steve R. Gunn, modified by Sunil Patel: www.sunilpatel.co.uk
%% Further modified by www.latextemplates.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\ProvidesClass{Thesis}
[2007/22/02 v1.0
LaTeX document class]
\def\baseclass{book}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\baseclass}}
\def\@checkoptions#1#2{
	\edef\@curroptions{\@ptionlist{\@currname.\@currext}}
	\@tempswafalse
	\@tfor\@this:=#2\do{
		\@expandtwoargs\in@{,\@this,}{,\@curroptions,}
		\ifin@ \@tempswatrue \@break@tfor \fi}
	\let\@this\@empty
	\if@tempswa \else \PassOptionsToClass{#1}{\baseclass}\fi
}
\@checkoptions{13pt}{{10pt}{11pt}{12pt}{13pt}}
\PassOptionsToClass{a4paper}{\baseclass}
\ProcessOptions\relax
\LoadClass{\baseclass}
\newcommand\bhrule{\typeout{------------------------------------------------------------------------------}}





\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}
\usepackage{epstopdf}
\usepackage{booktabs}
%\usepackage{hyperref}
\usepackage{rotating}
\usepackage{setspace}
\usepackage{vmargin}
\usepackage{todonotes}
\usepackage{wrapfig}
%\usepackage[fontsize=13pt]{scrextend}

\usepackage{color}
\definecolor{deepblue}{rgb}{0,0,0.5}
\definecolor{deepred}{rgb}{0.6,0,0}
\definecolor{deepgreen}{rgb}{0,0.5,0}

\usepackage{fancyhdr}
\lhead[\rm\thepage]{\fancyplain{}{\sl{\rightmark}}}
\rhead[\fancyplain{}{\sl{\leftmark}}]{\rm\thepage}
\chead{}\lfoot{}\rfoot{}\cfoot{}

\usepackage{listings}
\lstset{ 
	language=Python, 
	basicstyle=\footnotesize,        
	breakatwhitespace=false,         
	breaklines=true,                 
	commentstyle=\color{gray}, 	 
	identifierstyle=\color{deepblue}, 
	keywordstyle=\color{deepred},
	stringstyle=\color{deepgreen},   
	frame=tb,				 	               
	keepspaces=true,                 
	morekeywords={*,...},            
	%numbers=left,                    
	%numbersep=10pt,                   
	%numberstyle=\color{gray},   
	showspaces=false,                 
	showstringspaces=false,          
	showtabs=false,                  
	tabsize=2                     
	%title=\lstname   
	%xleftmargin=\parindent              
}

%\usepackage[centerlast,small,sc]{caption}

\usepackage[pdfpagemode={UseOutlines},bookmarks=true,bookmarksopen=true,
   bookmarksopenlevel=0,bookmarksnumbered=true,hypertexnames=false,
   colorlinks,linkcolor={blue},citecolor={blue},urlcolor={red},
   pdfstartview={FitV},unicode,breaklinks=true]{hyperref}
\pdfstringdefDisableCommands{
   \let\\\space
}

%\usepackage[square, numbers, comma, sort&compress]{natbib} % Use the natbib reference package - read up on this to edit the reference style; if you want text (e.g. Smith et al., 2012) for the in-text references (instead of numbers), remove 'numbers' 

\hypersetup{urlcolor=blue, colorlinks=true} % Colors hyperlinks in blue - change to black if annoying

%----------------------------------------------------------------------------------------
%	DOCUMENT VARIABLES
%	Fill in the lines below to update the thesis template
%	If you wish to cite each of the variables defined below, look at the
%	section above for the citation command e.g. \examiner{} below is
%	defined as \examname above so you cite it as \examname
%----------------------------------------------------------------------------------------

\newcommand*{\supervisor}[1]{\def\supname{#1}}
\newcommand*{\thesistitle}[1]{\def\ttitle{#1}}
\newcommand*{\degree}[1]{\def\degreename{#1}}
\newcommand*{\authors}[1]{\def\authornames{#1}}
\newcommand*{\authorsId}[1]{\def\authorId{#1}}
\newcommand*{\addresses}[1]{\def\addressnames{#1}}
\newcommand*{\university}[1]{\def\univname{#1}}
\newcommand*{\UNIVERSITY}[1]{\def\UNIVNAME{#1}}

\newcommand*{\institute}[1]{\def\instname{#1}}
\newcommand*{\INSTITUTE}[1]{\def\INSTNAME{#1}}

\newcommand*{\department}[1]{\def\deptname{#1}}
\newcommand*{\DEPARTMENT}[1]{\def\DEPTNAME{#1}}


\newcommand*{\faculty}[1]{\def\facname{#1}}
\newcommand*{\FACULTY}[1]{\def\FACNAME{#1}}
\newcommand*{\subject}[1]{\def\subjectname{#1}}
\newcommand*{\keywords}[1]{\def\keywordnames{#1}}


% Your thesis title - this is used in the title and abstract
\thesistitle{Single Element Feedforward Neural Network Inversion	Methods	in Python} 
%-------------------------------------------------  
% You supervisor's name - this is used in the title page
\supervisor{
\texorpdfstring{\href{mailto:bogdandy@iit.uni-miskolc.hu} 
				{Bence Bogdándy}}
				{Bence Bogdándy}
} 
%-------------------------------------------------   
% Your degree name - this is currently used in the title page and abstract
\degree{Bachelor's Thesis} 
%-------------------------------------------------   
% Your name - this is used in the title page and abstract
\authors{
\texorpdfstring{\href{mailto:juhasz40@iit.uni-miskolc.hu} 
                {Lilla Juhász}}
                {FWIY50} 
} 
%-------------------------------------------------   
% Neptun kod
\authorsId{FWIY50} 
%-------------------------------------------------   
\addresses{} % Your address - this is not currently used anywhere in the template, cite it with \addressnames if you want it
%-------------------------------------------------   
\subject{} % Your subject area - this is not currently used anywhere in the template, cite it with \subjectname if you want it
%-------------------------------------------------   
\keywords{} % Keywords for your thesis - this is not currently used anywhere in the template, cite it with \keywordnames if you want it
%-------------------------------------------------   
\university{\texorpdfstring{\href{http://www.uni-miskolc.hu} % Your university's URL
                {University of Miskolc}} % Your university's name - this is currently used in the title page
                {University of Miskolc}}
%-------------------------------------------------   
\UNIVERSITY{\texorpdfstring{\href{http://www.uni-miskolc.hu} % Your university's URL
                {UNIVERSITY OF MISKOLC}} % Your university's name in capitals - this is currently used in the abstract page
	                {UNIVERSITY OF MISKOLC}}
                
\institute{\texorpdfstring{\href{http://www.iit.uni-miskolc.hu} % Your university's URL
                {Institute of Information Science}} % Your university's name in capitals - this is currently used in the abstract page
                {Institute of Information Science}}
                
\INSTITUTE{\texorpdfstring{\href{http://www.iit.uni-miskolc.hu} % Your university's URL
                {INSTITUTE OF INFORMATION SCIENCE}} % Your university's name in capitals - this is currently used in the abstract page
                {INSTITUTE OF INFORMATION SCIENCE}}
%-------------------------------------------------                
\department{
	\texorpdfstring{
		\href{http://www.iit.uni-miskolc.hu} % Your department's URL
       {Department of Information Technology}
   } % Your department's name - used in the title page and abstract
   {Department of Information Technology}
}
%-------------------------------------------------                
\DEPARTMENT{\texorpdfstring{\href{Department or School Web Site URL Here (include http://)} % Your department's URL
                {DEPARTMENT OF INFORMATION TECHNOLOGY)}} % Your department's name in capitals - this is not currently used anywhere in the template, cite it with \DEPTNAME if you want it
                {DEPARTMENT OF INFORMATION TECHNOLOGY}}
%-------------------------------------------------                

%-------------------------------------------------                
\faculty{\texorpdfstring{\href{Faculty Web Site URL Here (include http://)} % Your faculty's URL
                {Faculty of Mechanical Engineering and Informatics }} % Your faculty's name - this is currently used in the abstract page
                {Faculty of Mechanical Engineering and Informatics}}
%-------------------------------------------------                
\FACULTY{\texorpdfstring{\href{Faculty Web Site URL Here (include http://)} % Your faculty's URL
                {FACULTY OF MECHANICAL ENGINEERING AND INFORMATICS}} % Your faculty's name in capitals - this is not currently used anywhere in the template, cite it with \FACNAME if you want it
                {FACULTY OF MECHANICAL ENGINEERING AND INFORMATICS}}
%----------------------------------------------------------------------------------------    

% The default title page, this is not used in this www.latextemplates.com version            

