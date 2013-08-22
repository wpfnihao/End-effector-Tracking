if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
imap <S-Tab> <Plug>SuperTabBackward
inoremap <C-Tab> 	
imap <silent> <Plug>IMAP_JumpBack =IMAP_Jumpfunc('b', 0)
imap <silent> <Plug>IMAP_JumpForward =IMAP_Jumpfunc('', 0)
map! <S-Insert> <MiddleMouse>
nmap <silent>  :nbkey C-B
nmap <silent>  :nbkey C-D
nmap <silent>  :nbkey C-E
noremap  h
noremap <NL> j
noremap  k
noremap  l
nmap <silent>  :nbkey C-N
nmap <silent>  :nbkey C-P
nmap <silent>  :nbkey C-U
nmap <silent>  :nbkey C-X
nmap <silent>  :nbkey C-Z
nmap d :cs find d =expand("<cword>")	
nmap i :cs find i ^=expand("<cfile>")$
nmap f :cs find f =expand("<cfile>")	
nmap e :cs find e =expand("<cword>")	
nmap t :cs find t =expand("<cword>")	
nmap c :cs find c =expand("<cword>")	
nmap g :cs find g =expand("<cword>")	
nmap s :cs find s =expand("<cword>")	
nmap <silent> A :nbkey S-A
nmap <silent> B :nbkey S-B
nmap <silent> C :nbkey S-C
nmap <silent> F :nbkey S-F
nmap <silent> L :nbkey S-L
nmap <silent> Q :nbkey S-Q
nmap <silent> R :nbkey S-R
nmap <silent> S :nbkey S-S
nmap <silent> W :nbkey S-W
nmap <silent> X :nbkey S-X
map \mbt <Plug>TMiniBufExplorer
map \mbu <Plug>UMiniBufExplorer
map \mbc <Plug>CMiniBufExplorer
map \mbe <Plug>MiniBufExplorer
nmap \ihn :IHN
nmap \is :IHS:A
nmap \ih :IHS
nmap gx <Plug>NetrwBrowseX
nnoremap j gj
nnoremap k gk
map <silent> mm <Plug>Vm_toggle_sign 
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#NetrwBrowseX(expand("<cWORD>"),0)
map <S-F2> <Plug>Vm_goto_prev_sign
map <F2> <Plug>Vm_goto_next_sign
map <C-F2> <Plug>Vm_toggle_sign
noremap <C-Right> l
noremap <C-Left> h
noremap <C-Up> k
noremap <C-Down> j
vmap <silent> <Plug>IMAP_JumpBack `<i=IMAP_Jumpfunc('b', 0)
vmap <silent> <Plug>IMAP_JumpForward i=IMAP_Jumpfunc('', 0)
vmap <silent> <Plug>IMAP_DeleteAndJumpBack "_<Del>i=IMAP_Jumpfunc('b', 0)
vmap <silent> <Plug>IMAP_DeleteAndJumpForward "_<Del>i=IMAP_Jumpfunc('', 0)
nmap <silent> <Plug>IMAP_JumpBack i=IMAP_Jumpfunc('b', 0)
nmap <silent> <Plug>IMAP_JumpForward i=IMAP_Jumpfunc('', 0)
nmap <Nul><Nul>d :vert scs find d =expand("<cword>")
nmap <Nul><Nul>i :vert scs find i ^=expand("<cfile>")$	
nmap <Nul><Nul>f :vert scs find f =expand("<cfile>")	
nmap <Nul><Nul>e :vert scs find e =expand("<cword>")
nmap <Nul><Nul>t :vert scs find t =expand("<cword>")
nmap <Nul><Nul>c :vert scs find c =expand("<cword>")
nmap <Nul><Nul>g :vert scs find g =expand("<cword>")
nmap <Nul><Nul>s :vert scs find s =expand("<cword>")
nmap <Nul>d :scs find d =expand("<cword>")	
nmap <Nul>i :scs find i ^=expand("<cfile>")$	
nmap <Nul>f :scs find f =expand("<cfile>")	
nmap <Nul>e :scs find e =expand("<cword>")	
nmap <Nul>t :scs find t =expand("<cword>")	
nmap <Nul>c :scs find c =expand("<cword>")	
nmap <Nul>g :scs find g =expand("<cword>")	
nmap <Nul>s :scs find s =expand("<cword>")	
nmap <F9> :UpdateTypesFile 
nmap <F8> :TagbarToggle 
map <S-Insert> <MiddleMouse>
imap 	 <Plug>SuperTabForward
imap <NL> <Plug>IMAP_JumpForward
imap  <Plug>SuperTabForward
imap  <Plug>SuperTabBackward
inoremap <expr>  omni#cpp#maycomplete#Complete()
inoremap <expr> . omni#cpp#maycomplete#Dot()
inoremap <expr> : omni#cpp#maycomplete#Scope()
inoremap <expr> > omni#cpp#maycomplete#Arrow()
imap \ihn :IHN
imap \is :IHS:A
imap \ih :IHS
let &cpo=s:cpo_save
unlet s:cpo_save
set backspace=indent,eol,start
set balloondelay=100
set completeopt=menu,menuone
set cscopetag
set cscopeverbose
set noequalalways
set fileencodings=utf-8,gb2312,gbk,gb18030
set grepprg=grep\ -nH\ $*
set helplang=cn
set history=50
set hlsearch
set langmenu=zh_CN.UTF-8
set nomodeline
set mouse=a
set omnifunc=omni#cpp#complete#Main
set path=.,/usr/include,,,/opt/ros/fuerte/stacks/vision_visp/visp/install/include,/opt/ros/fuerte/include,~/fuerte_workspace/endeffector_tracking/include
set printoptions=paper:a4
set ruler
set runtimepath=~/.vim,/var/lib/vim/addons,/usr/share/vim/vimfiles,/usr/share/vim/vim73,/usr/share/vim/vimfiles/after,/var/lib/vim/addons/after,~/.vim/after
set shiftwidth=4
set spellfile=~/.vim/dict.add
set suffixes=.bak,~,.swp,.o,.info,.aux,.log,.dvi,.bbl,.blg,.brf,.cb,.ind,.idx,.ilg,.inx,.out,.toc
set tabstop=4
set tags=/opt/ros/fuerte/stacks/camera_umd/uvc_camera/tags,/opt/ros/fuerte/stacks/vision_visp/visp/install/include/visp/tags,/opt/ros/fuerte/include/opencv2/tags,/usr/include/c++/4.6.3/tags,./tags,~/.vimtags
set termencoding=utf-8
set window=48
" vim: set ft=vim :
