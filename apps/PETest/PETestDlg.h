
// PETestDlg.h: 헤더 파일
//

#pragma once

#include <sld_threadpool_manager.h>
#include <sld_ff_demuxer.h>
#include <sld_nvdecoder.h>
#include <sld_pose_estimator.h>
#include <sld_nvrenderer.h>


// CPETestDlg 대화 상자
class CPETestDlg
	: public CDialogEx
	, public solids::lib::misc::threadpool::manager
	, public solids::lib::container::ff::demuxer
{
// 생성입니다.
public:
	CPETestDlg(CWnd* pParent = nullptr);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_PETEST_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.

	virtual void on_video_begin(int32_t codec, const uint8_t* extradata, int32_t extradataSize, int32_t width, int32_t height, double fps);
	virtual void on_video_recv(uint8_t* bytes, int32_t nbytes, int32_t nFrameIdx);
	virtual void on_video_end(void);

private:
	solids::lib::video::nvidia::decoder::context_t _decoder_ctx;
	solids::lib::video::nvidia::decoder* _decoder;

	solids::lib::video::nvidia::pose::estimator::context_t _estimator_ctx;
	solids::lib::video::nvidia::pose::estimator* _estimator;

	solids::lib::video::nvidia::renderer::context_t _renderer_ctx;
	solids::lib::video::nvidia::renderer* _renderer;
// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	virtual BOOL DestroyWindow();
	afx_msg void OnBnClickedButtonPlay();
	afx_msg void OnBnClickedButtonStop();
};
