
// SLDPlayerDlg.cpp : implementation file
//

#include "pch.h"
#include "framework.h"
#include "SLDPlayer.h"
#include "SLDPlayerDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
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


// CSLDPlayerDlg dialog



CSLDPlayerDlg::CSLDPlayerDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_SLDPLAYER_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CSLDPlayerDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CSLDPlayerDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_PLAY, &CSLDPlayerDlg::OnBnClickedButtonPlay)
	ON_BN_CLICKED(IDC_BUTTON_STOP, &CSLDPlayerDlg::OnBnClickedButtonStop)
END_MESSAGE_MAP()


// CSLDPlayerDlg message handlers

BOOL CSLDPlayerDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
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

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	//ShowWindow(SW_MAXIMIZE);
	//ShowWindow(SW_MINIMIZE);

	// TODO: Add extra initialization here

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CSLDPlayerDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CSLDPlayerDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CSLDPlayerDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CSLDPlayerDlg::OnBnClickedButtonPlay()
{
	// TODO: Add your control notification handler code here
	HWND videoHwnd = ::GetDlgItem(GetSafeHwnd(), IDC_STATIC_VIDEO_VIEW);
	_context.hwnd = videoHwnd;
	_context.repeat = TRUE;
	_context.userdata = NULL;

	//::wcsncpy_s(_context.url, L"rtsp://192.168.0.43/samsung.mkv", MAX_PATH); //P010
	//::wcsncpy_s(_context.url, L"rtsp://192.168.0.43/costarica.mkv", MAX_PATH); //NV12
	//::wcsncpy_s(_context.url, L"rtsp://192.168.0.43/vod/costarica_hevc.mp4", MAX_PATH); //NV12
	::wcsncpy_s(_context.url, L"rtsp://192.168.0.43/vod/FHD_AVC.mp4", MAX_PATH); //NV12
	//::wcsncpy_s(_context.url, L"rtsp://192.168.0.43/vod/test_sample.mp4", MAX_PATH); //NV12
	//::wcsncpy_s(_context.url, L"rtsp://192.168.0.43/costarica_avc.mkv", MAX_PATH);
	//::wcsncpy_s(_context.url, L"rtsp://192.168.56.1:554/samsung.mkv", MAX_PATH);
	//::wcsncpy_s(_context.url, L"rtsp://192.168.0.43/FHD_AVC.mkv", MAX_PATH);
	//::wcsncpy_s(_context.url, L"rtsp://192.168.18.119:8554/main", MAX_PATH);
	//::wcsncpy_s(_context.url, L"rtsp://192.168.18.30:8554/4DReplayLive?type=FHD", MAX_PATH);
	//::wcsncpy_s(_context.url, L"F:\\workspace\\reference\\contents\\FHD_AVC.mp4", MAX_PATH);
	//::wcsncpy_s(_context.url, L"F:\\workspace\\4dreplay\\rtsp_server\\samsung.mp4", MAX_PATH);
	//::wcsncpy_s(_context.url, L"F:\\workspace\\reference\\contents\\costarica_hevc.mp4", MAX_PATH);

	_controller.open(&_context);
	_controller.play();
}

void CSLDPlayerDlg::OnBnClickedButtonStop()
{
	// TODO: Add your control notification handler code here
	_controller.stop();
	_controller.close();
}
