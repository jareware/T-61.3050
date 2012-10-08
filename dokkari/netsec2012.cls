% Written by Pasi Eronen (pe at iki dot fi) for T-110.551 spring 2004
% Changes for Spring 2005 made by Nadja Kasinskaja, 2005-01-12
% Changes for Spring 2006 made by Sanna Liimatainen, 2006-01-12
% Changes for Autumn 2007 made by Laura Takkinen, 2007-09-17
% Changes for Autumn 2008 made by Jani Heikkinen, 2008-08-22
% Changes for Spring 2010 made by Sachin Gaur, 2009-12-14
% Changes for Spring 2011, Spring 2012, Autumn 2012 made by Sandeep Tamrakar
% Updates for Autumn 2012 made by Mario Di Francesco

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{netsec2012}[2012/01/16 Aalto University T-61.3050 Machine Learning: Basic Introduction]

\newif\if@cameraready
\@camerareadyfalse
\DeclareOption{cameraready}{\@camerareadytrue}
\ProcessOptions

% Basic stuff
\if@cameraready
  \LoadClass[10pt,twocolumn,a4paper]{article}
\else
  \LoadClass[10pt, a4paper]{article}
\fi
\RequirePackage{times}
\RequirePackage[T1]{fontenc}
\RequirePackage{graphicx}
\RequirePackage{fancyhdr}
\RequirePackage{url}
\RequirePackage{setspace}

% Adjust margins
\addtolength{\textwidth}{14mm}
\addtolength{\hoffset}{-7mm}
\addtolength{\textheight}{45mm}
\addtolength{\voffset}{-18mm}

% Abbreviated bibliography style
\bibliographystyle{abbrv}

% Line spacing for draft and final papers
\if@cameraready
  \singlespacing
\else
  \doublespacing
\fi
 
%isodate from manuals of University of Oulu, time removed
%http://www.oulu.fi/atkk/tkpalv/unix/isodate.html
%referred 2004-02-18
%added and dates in the document changed by Nadja Kasinskaja

\newcommand{\twodigits}[1]{\ifnum #1<10 0\fi \number#1}

\newcommand{\isodate}{%
  \the\year-\twodigits{\month}-\twodigits{\day}}

%endof isodate

\pagestyle{fancyplain}
\addtolength{\headsep}{-10pt}
\fancyhead{}
\fancyfoot{}
\fancyhead[LO]{\fancyplain{}{Aalto University T-61.3050 Machine Learning: Basic Introduction}}
\if@cameraready
  \fancyhead[RO]{\fancyplain{}{Autumn 2012}}
\else
  \fancyhead[RO]{\fancyplain{}
                 %%{Draft {\number\day}.{\number\month}.{\number\year}}}
                 {Draft \isodate}}    
  \fancyfoot[CO]{\thepage}
\fi
\date{}
