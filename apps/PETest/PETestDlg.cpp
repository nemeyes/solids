
// PETestDlg.cpp: 구현 파일
//

#include "pch.h"
#include "framework.h"
#include "PETest.h"
#include "PETestDlg.h"
#include "afxdialogex.h"

#include <sld_stringhelper.h>
#include "on_video_end_worker.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CPETestDlg 대화 상자



CPETestDlg::CPETestDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_PETEST_DIALOG, pParent)
	, _decoder(NULL)
	, _estimator(NULL)
	, _renderer(NULL)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CPETestDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CPETestDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_PLAY, &CPETestDlg::OnBnClickedButtonPlay)
	ON_BN_CLICKED(IDC_BUTTON_STOP, &CPETestDlg::OnBnClickedButtonStop)
END_MESSAGE_MAP()


// CPETestDlg 메시지 처리기

BOOL CPETestDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.
	solids::lib::misc::threadpool::manager::initialize(15);
	solids::lib::misc::threadpool::manager::add_worker(std::shared_ptr<solids::lib::misc::threadpool::worker>(new on_video_end_worker(this)));

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CPETestDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 애플리케이션의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CPETestDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CPETestDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

BOOL CPETestDlg::DestroyWindow()
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.
	solids::lib::misc::threadpool::manager::release();

	return __super::DestroyWindow();
}


void CPETestDlg::OnBnClickedButtonPlay()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	CFileDialog dlg(TRUE, L"mp4", NULL, NULL,
		L"MP4 Files (*.mp4)|*.mp4|"
		L"Mastroka Files (*.mkv)|*.mkv|"
		L"AVI Files (*.avi)|*.avi|"
		L"All Files||");
	if (dlg.DoModal() == IDOK)
	{
		CString filePath = dlg.GetPathName();

		char* container = NULL;
		solids::lib::stringhelper::convert_wide2multibyte((LPWSTR)(LPCWSTR)filePath, &container);
		if (container)
		{
			if (solids::lib::container::ff::demuxer::play(container) == solids::lib::container::ff::demuxer::err_code_t::success)
			{
				CWnd* wnd = GetDlgItem(IDC_BUTTON_PLAY);
				wnd->EnableWindow(FALSE);
				wnd = GetDlgItem(IDC_BUTTON_STOP);
				wnd->EnableWindow(TRUE);
			}
			free(container);
			container = NULL;
		}
	}
}


void CPETestDlg::OnBnClickedButtonStop()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	solids::lib::container::ff::demuxer::stop();
	CWnd* wnd = GetDlgItem(IDC_BUTTON_STOP);
	wnd->EnableWindow(FALSE);
	wnd = GetDlgItem(IDC_BUTTON_PLAY);
	wnd->EnableWindow(TRUE);
}

void CPETestDlg::on_video_begin(int32_t codec, const uint8_t* extradata, int32_t extradataSize, int32_t width, int32_t height, double fps)
{
	if (_decoder)
	{
		if (_decoder->is_initialized())
			_decoder->release();
		delete _decoder;
		_decoder = NULL;
	}

	if (_renderer)
	{
		if (_renderer->is_initialized())
			_renderer->release();
		delete _renderer;
		_renderer = NULL;
	}

	_decoder = new solids::lib::video::nvidia::decoder();
	_decoder_ctx.codec = codec;
	_decoder_ctx.width = width;
	_decoder_ctx.height = height;
	_decoder_ctx.colorspace = solids::lib::video::nvidia::decoder::colorspace_t::bgra;
	_decoder->initialize(&_decoder_ctx);

	int32_t nDecoded = 0;
	uint8_t** ppDecoded = NULL;
	long long* pTimestamp = NULL;
	_decoder->decode((uint8_t*)extradata, extradataSize, 0, &ppDecoded, &nDecoded, &pTimestamp);


	_estimator = new solids::lib::video::nvidia::pose::estimator();
	_estimator_ctx.width = width;
	_estimator_ctx.height = height;
	_estimator->initialize(&_estimator_ctx);

	_renderer = new solids::lib::video::nvidia::renderer();
	_renderer_ctx.cuctx = _decoder->context();
	_renderer_ctx.width = width;
	_renderer_ctx.height = height;
	_renderer_ctx.nstaging = 1;
	_renderer_ctx.hwnd = ::GetDlgItem(this->GetSafeHwnd(), IDC_STATIC_VIDEO_VIEW);
	_renderer->initialize(&_renderer_ctx);

	CWnd* wnd = GetDlgItem(IDC_BUTTON_PLAY);
	wnd->EnableWindow(FALSE);
	wnd = GetDlgItem(IDC_BUTTON_STOP);
	wnd->EnableWindow(TRUE);
}

void CPETestDlg::on_video_recv(uint8_t* bytes, int32_t nbytes, int32_t nFrameIdx)
{
	int32_t nDecoded = 0;
	uint8_t** ppDecoded = NULL;
	long long* pTimestamp = NULL;
	_decoder->decode(bytes, nbytes, 0, &ppDecoded, &nDecoded, &pTimestamp);
	for (int i = 0; i < nDecoded; i++)
	{
		uint8_t* render = NULL;
		int32_t pitch = 0;
		_estimator->estimate(ppDecoded[i], (int32_t)_decoder->get_pitch2(), &render, pitch);
		_renderer->render(render, pitch);
	}
}

void CPETestDlg::on_video_end(void)
{
	if (_renderer)
		_renderer->release();
	if (_decoder)
		_decoder->release();
	if (_estimator)
		_estimator->release();
	if (_renderer)
		delete _renderer;
	if (_decoder)
		delete _decoder;
	if (_estimator)
		delete _estimator;
	_renderer = NULL;
	_decoder = NULL;
	_estimator = NULL;

	solids::lib::misc::threadpool::manager::run(on_video_end_worker::VIDEO_END_WORKER_ID, NULL, 0, this);
}